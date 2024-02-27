# Training

For single gpu, or debugging purposes, use 
`python train_image_variation.py`
## Parallel

For multi-gpu training, before starting, please run 
`accelerate config` to set parallel configurations. Or you can directly go to `/pfs/mt-1oY5F7/luoyihao/.cache/huggingface/accelerate/default_config.yaml` to change them. 

Recommended:

In which compute environment are you running? 
This machine                                                      
Which type of machine are you using?
Multi-gpu                                                                                                                                                                               
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                      
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: yes                                                                                                                              
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                               
Do you want to use DeepSpeed? [yes/NO]: NO                                                                                      
Do you want to use FullyShardedDataParallel? [yes/NO]: NO                                                                      
Do you want to use Megatron-LM ? [yes/NO]: NO

**Check available gpus first and then set them, you may need to change numebr of gpus and gpu list while training**

How many GPU(s) should be used for distributed training? [1]:4

What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:0,1,2,3

Do you wish to use FP16 or BF16 (mixed precision)?
fp16                                                                                                                            
After setting everything, start train with `accelerate launch train_image_variation.py`

## Training Configs
All training configs are stored in 'GNN3D/configs/training_config'.

Notes:
- Before training or debugging, please change the `wandb_args`'s entity to yours in training configs, logging and validation part.
- For training code with annotations, refer to `train_image_variation.py` from row 502 in `main()` function.


# Datasets
Refer to `GNN3D/data/mvds.py`, which contains a dataset class for multi-view diffusion.

For each batch, the dataset should return 
- image_cond: `[bsz, channel, height, width]` The raw image which the model conditions on
- image_target: `[bsz, num_views, channel, height, width]` The raw image which the model wish to generate, i.e y_pred
- normal_target: `[bsz, num_views, channel, height, width]` The raw normal image which the model wish to generate, i.e. y_pred in another domain.
- image_cond_vae: `[bsz, channel, height, width]` The processed image that are ready for vae to process.
- image_cond_CLIP: `[bsz, channel, height, width]` The processed image that are ready for CLIPImageProcessor to proecess.
- image_target_vae: `[bsz, num_views, channel, height, width]` The processed image that are ready for vae to encode.
- normal_target_vae: `[bsz, num_views, channel, height, width]` The processed normal image that are ready for vae to encode.
- camera_embed_image: `[bsz, num_views, 10]` The embedded camera_embedding and domain switcher for the image data. (Has length 10 according to wonder3d)
- camera_embed_normal: `[bsz, num_views, 10]` The embedded camera_embedding and domain switcher for normal data.





## Num_views setting and batch_size setting


Since attention is performed inside num_views data, the real batch size is larger than the setting. For example, if `num_views = 6`, and batch size is set to 2. Then the real batch size fed into network is `num_views*2*batch_size`, in order to be compatible with `cd_attention_mid` or `cd_attention_last` is `True` in unet's config.



