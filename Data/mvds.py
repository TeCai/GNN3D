from torch.utils.data import Dataset
from pathlib import Path
import torch


class MVDataset_debug(Dataset):
    def __init__(self, data_path = Path('/.'), random_seed = 42):
        self.image_cond = torch.randn(size = (101,3,256,256))
        self.image_target = torch.randn(size = (101,3,256,256))
        self.camera_pose = torch.randn(size = (101,3))
        self.task_label = torch.distributions.one_hot_categorical.OneHotCategorical(probs=torch.tensor(
            [0.3,0.3,0.4])).sample(sample_shape=(101,))


    def __len__(self):
        return 101

    def __getitem__(self, idx):
        return  {
            'image_cond': self.image_cond[idx],
            'image_target': self.image_target[idx],
            'camera_pose': self.camera_pose[idx],
            'task_label': self.task_label[idx]
        }

