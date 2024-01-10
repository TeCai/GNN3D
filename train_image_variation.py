# importing

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import yaml

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.utils import ContextManagers
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

import diffusers
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- image-variation
- diffusers
inference: true
---
    """
    model_card = f"""
# image-variation finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)



def log_validation(vae, image_encoder, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = MVDiffusionImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        image_encoder=accelerator.unwrap_model(image_encoder),
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        feature_extractor=None,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

# TODO: validation pipeline

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

# end TODO

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images




def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default='configs/tran_config.yml',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    args = parser.parse_args()
    cfg = yaml.safe_load(args.config_path)
    cfg.update{'config_path': args.config_path}
    args = cfg

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args



def main():

    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = KarrasDiffusionSchedulers.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # TODO: tobe continued



    # parser.add_argument(
    #     "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    # )
    # parser.add_argument(
    #     "--pretrained_model_name_or_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to pretrained model or model identifier from huggingface.co/models.",
    # )
    # parser.add_argument(
    #     "--revision",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="Revision of pretrained model identifier from huggingface.co/models.",
    # )
    # parser.add_argument(
    #     "--variant",
    #     type=str,
    #     default=None,
    #     help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    # )
    #
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default='./',
    #     help="Path containing data that can be processed by mvds dataset."
    # )
    #
    #
    # parser.add_argument(
    #     "--max_train_samples",
    #     type=int,
    #     default=None,
    #     help=(
    #         "For debugging purposes or quicker training, truncate the number of training examples to this "
    #         "value if set."
    #     ),
    # )
    # parser.add_argument(
    #     "--validation_prompts",
    #     type=str,
    #     default=None,
    #     nargs="+",
    #     help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="sd-model-finetuned",
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    # parser.add_argument(
    #     "--cache_dir",
    #     type=str,
    #     default=None,
    #     help="The directory where the downloaded models and datasets will be stored.",
    # )
    # parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--resolution",
    #     type=int,
    #     default=256,
    #     help=(
    #         "The resolution for input images, all the images in the train/validation dataset will be resized to this"
    #         " resolution"
    #     ),
    # )
    # parser.add_argument(
    #     "--center_crop",
    #     default=False,
    #     action="store_true",
    #     help=(
    #         "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
    #         " cropped. The images will be resized to the resolution first before cropping."
    #     ),
    # )
    # parser.add_argument(
    #     "--random_flip",
    #     default = False,
    #     action="store_true",
    #     help="whether to randomly flip images horizontally",
    # )
    # parser.add_argument(
    #     "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    # )
    # parser.add_argument("--num_train_epochs", type=int, default=100)
    # parser.add_argument(
    #     "--max_train_steps",
    #     type=int,
    #     default=None,
    #     help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    # )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument(
    #     "--gradient_checkpointing",
    #     action="store_true",
    #     help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    # )
    # parser.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     default=1e-4,
    #     help="Initial learning rate (after the potential warmup period) to use.",
    # )
    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    # parser.add_argument(
    #     "--lr_scheduler",
    #     type=str,
    #     default="constant",
    #     help=(
    #         'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
    #         ' "constant", "constant_with_warmup"]'
    #     ),
    # )
    # parser.add_argument(
    #     "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    # )
    # parser.add_argument(
    #     "--snr_gamma",
    #     type=float,
    #     default=None,
    #     help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
    #     "More details here: https://arxiv.org/abs/2303.09556.",
    # )
    # parser.add_argument(
    #     "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    # )
    # # parser.add_argument(
    # #     "--allow_tf32",
    # #     action="store_true",
    # #     help=(
    # #         "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
    # #         " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
    # #     ),
    # # )
    # parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    # parser.add_argument(
    #     "--non_ema_revision",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help=(
    #         "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
    #         " remote repository specified with --pretrained_model_name_or_path."
    #     ),
    # )
    # parser.add_argument(
    #     "--dataloader_num_workers",
    #     type=int,
    #     default=0,
    #     help=(
    #         "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
    #     ),
    # )
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    # parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    # parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    # parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    # parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    # parser.add_argument(
    #     "--prediction_type",
    #     type=str,
    #     default=None,
    #     help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    # )
    # parser.add_argument(
    #     "--hub_model_id",
    #     type=str,
    #     default=None,
    #     help="The name of the repository to keep in sync with the local `output_dir`.",
    # )
    # parser.add_argument(
    #     "--logging_dir",
    #     type=str,
    #     default="logs",
    #     help=(
    #         "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
    #         " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    #     ),
    # )
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     default=None,
    #     choices=["no", "fp16", "bf16"],
    #     help=(
    #         "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
    #         " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
    #         " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    #     ),
    # )
    # parser.add_argument(
    #     "--report_to",
    #     type=str,
    #     default="tensorboard",
    #     help=(
    #         'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
    #         ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
    #     ),
    # )
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument(
    #     "--checkpointing_steps",
    #     type=int,
    #     default=500,
    #     help=(
    #         "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
    #         " training using `--resume_from_checkpoint`."
    #     ),
    # )
    # parser.add_argument(
    #     "--checkpoints_total_limit",
    #     type=int,
    #     default=None,
    #     help=("Max number of checkpoints to store."),
    # )
    # parser.add_argument(
    #     "--resume_from_checkpoint",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Whether training should be resumed from a previous checkpoint. Use a path saved by"
    #         ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    #     ),
    # )
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    # )
    # parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    # parser.add_argument(
    #     "--validation_epochs",
    #     type=int,
    #     default=5,
    #     help="Run validation every X epochs.",
    # )
    # parser.add_argument(
    #     "--tracker_project_name",
    #     type=str,
    #     default="image2image-fine-tune",
    #     help=(
    #         "The `project_name` argument passed to Accelerator.init_trackers for"
    #         " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
    #     ),
    # )