from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch


class MVDataset(Dataset):
    def __init__(self, args, Processor: lambda x:x):
        self.args = args
        
        if self.args.debug:
            self.image_cond = torch.randint(low = 0, high = 255, size = (300,3,256,256),dtype=torch.uint8)
            self.image_target = torch.randint(low = 0, high = 255, size = (300,3,256,256),dtype=torch.uint8)
            # self.image_cond = [transforms.ToPILImage()(image) for image in image_cond]
            # self.image_target = [transforms.ToPILImage()(image) for image in image_target]
            self.camera_pose = torch.randn(size = (300,10))
            # self.domain_switcher = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.tensor(
            #     [0.3,0.3,0.4])).sample(sample_shape=(300,))
            self.Processor = Processor
            self.data_len = 300
        else:
            raise NotImplementedError
        
        assert self.data_len % (args.num_views*2) == 0, f'The length of data_len {self.data_len} must devide num_views{args.num_views}*2 '
        

    def __len__(self):
        return 300 if self.args.debug else None

    def __getitem__(self, idx):
        return  {
            'image_cond': self.image_cond[idx],
            'image_target': self.image_target[idx],
            'image_cond_vae': self.prepare_latents(transforms.ToPILImage()(self.image_cond[idx])),
            'image_cond_CLIP': self.prepare_embeds(transforms.ToPILImage()(self.image_cond[idx])),
            'image_target_vae': self.prepare_latents(transforms.ToPILImage()(self.image_target[idx])),
            'camera_embed': self.camera_pose[idx],
        } if self.args.debug else None

    def prepare_latents(self, image):
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution) if self.args.center_crop else transforms.RandomCrop(self.args.resolution),
                transforms.RandomHorizontalFlip() if self.args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return train_transforms(image)

    def prepare_embeds(self, image):
        image_pt = self.Processor(images=image, return_tensors="pt").pixel_values
        return image_pt.squeeze()

