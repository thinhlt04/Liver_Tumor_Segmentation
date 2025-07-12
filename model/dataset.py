import torch
from torch.utils.data import Dataset, DataLoader
import os 
import SimpleITK as sitk
import numpy as np

class LiTS(Dataset):
    def __init__(self, root, lowerbound, upperbound, train=False, dev=False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        if train:
            root = os.path.join(root, 'train')
        elif dev:
            root = os.path.join(root, 'dev')
        else:
            root = os.path.join(root, 'test')
        image_folder = os.path.join(root, 'image')
        target_folder = os.path.join(root, 'target')
        for image, target in zip(sorted(os.listdir(image_folder)), sorted(os.listdir(target_folder))):
            image_path = os.path.join(image_folder, image)
            target_path = os.path.join(target_folder, target)
            self.images.append(image_path)
            self.targets.append(target_path)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        
        target = sitk.ReadImage(self.targets[idx])
        target = sitk.GetArrayFromImage(target)
        # target[target == 1] = 0
        target[target == 2] = 1 
        target = target.astype(np.float32)
        
        clamp_filter = sitk.ClampImageFilter()
        clamp_filter.SetLowerBound(self.lowerbound)  
        clamp_filter.SetUpperBound(self.upperbound)
        
        clamped_image = clamp_filter.Execute(image)
        clamped_image = sitk.GetArrayFromImage(clamped_image)
        if self.transform:
            clamped_image = clamped_image.transpose(1, 2, 0)
            clamped_image = self.transform(clamped_image)
        if self.target_transform:
            target = target.transpose(1, 2, 0)
            target = self.target_transform(target)
        target = (target > 0).float()
        return clamped_image.float(), target