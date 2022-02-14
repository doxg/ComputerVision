import os
import pdb
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


if __name__ == "__main__":
    train_transforms = A.Compose([A.Rotate(limit=35, p=1.0),  #### Probability of applying transform = p
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.1),
                                  A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[1.0, 1.0, 1.0],
                                             max_pixel_value=255.0),
                                  ToTensorV2()])
    test_dataset = MyDataset(image_dir="data/train_images/", mask_dir="data/train_masks/", transform=None)
    pdb.set_trace()