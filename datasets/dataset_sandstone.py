import pdb
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
import glob
import os
from torch.utils.data import Dataset
import cv2
import numpy as np


def decompose_tif_images(path, set):
    image = Image.open(path)
    for i in range(1600):
        try:
            image.seek(i)
            image.save(
                f'C:/Users/user/PycharmProjects/Dastan/Unet/sandstone_data/decomposed/{set}/{set}{i}.decomposed.tif')
        except EOFError:
            break


class SandStoneDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super(SandStoneDataset, self).__init__()
        self.image_path = image_dir
        self.mask_path = mask_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_path, os.listdir(self.image_path)[index]), 0)
        image = np.array(image, dtype=np.float32)
        mask = np.array(cv2.imread(os.path.join(self.mask_path, os.listdir(self.mask_path)[index]), 0))
        mask = mask - 1  # starts from 0,1,2,3

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":
    train_path = "../sandstone_data/decomposed/"
    # Already decomposed
    # decompose_tif_images("../sandstone_data/train_images.tif", "train")
    # decompose_tif_images("../sandstone_data/mask_images.tif", "validation")

    train_transforms = A.Compose([A.Rotate(limit=35, p=1.0),  #### Probability of applying transform = p
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.1),
                                  A.Normalize(mean=0.0,
                                              std=1.0,
                                              max_pixel_value=255.0),
                                  ToTensorV2()])
    without_transforms = SandStoneDataset(image_dir="../sandstone_data/decomposed/train_images",
                                          mask_dir="../sandstone_data/decomposed/train_masks", transform=None)
    train_dataset = SandStoneDataset(image_dir="../sandstone_data/decomposed/train_images",
                                     mask_dir="../sandstone_data/decomposed/train_masks", transform=train_transforms)
