from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch


class MVDataset(Dataset):
    def __init__(self, args, Processor: lambda x:x):
        self.args = args
        
        if self.args.debug:
            self.image_cond = torch.randint(low = 0, high = 255, size = (300,3,256,256),dtype=torch.uint8)
            self.image_target = torch.randint(low = 0, high = 255, size = (300,args.num_views,3,256,256),dtype=torch.uint8)
            # self.image_cond = [transforms.ToPILImage()(image) for image in image_cond]
            # self.image_target = [transforms.ToPILImage()(image) for image in image_target]
            self.normal_target = torch.randint(low = 0, high = 255, size = (300,args.num_views,3,256,256),dtype=torch.uint8)
            self.camera_pose_image = torch.randn(size = (300,args.num_views,10))
            self.camera_pose_normal = torch.randn(size = (300,args.num_views,10))
            # self.domain_switcher = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.tensor(
            #     [0.3,0.3,0.4])).sample(sample_shape=(300,))
            self.Processor = Processor
            self.data_len = 300
        else:
            raise NotImplementedError
        
        assert self.data_len % (args.num_views) == 0, f'The length of data_len {self.data_len} must devide num_views{args.num_views} '
        

    def __len__(self):
        return 300 if self.args.debug else None

    def __getitem__(self, idx):
        return  {
            'image_cond': self.image_cond[idx],
            'image_target': self.image_target[idx],
            'normal_target': self.normal_target[idx],
            'image_cond_vae': self.prepare_latents(transforms.ToPILImage()(self.image_cond[idx])),
            'image_cond_CLIP': self.prepare_embeds(transforms.ToPILImage()(self.image_cond[idx])),
            'image_target_vae': self.prepare_latents([transforms.ToPILImage()(image) for image in self.image_target[idx]]),
            'normal_target_vae': self.prepare_latents([transforms.ToPILImage()(image) for image in self.normal_target[idx]]),
            'camera_embed_image': self.camera_pose_image[idx],
            'camera_embed_normal': self.camera_pose_normal[idx]
        } if self.args.debug else None

    def prepare_latents(self, image_list):
        train_transforms = transforms.Compose(
            [
                transforms.Resize(self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.args.resolution) if self.args.center_crop else transforms.RandomCrop(self.args.resolution),
                transforms.RandomHorizontalFlip() if self.args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # if type(image_list) is list:
        #     a = torch.concat([train_transforms(image) for image in image_list],dim=0)
        return torch.stack([train_transforms(image) for image in image_list],dim=0) if type(image_list) is list else train_transforms(image_list)

    def prepare_embeds(self, image):
        

        return self.Processor(images=image, return_tensors="pt").pixel_values.squeeze()

