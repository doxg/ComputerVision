import os
import pdb
import numpy as np


def separate(path):
    images = os.listdir(os.path.join(path, "images"))
    masks = os.listdir(os.path.join(path, "masks"))

    val_ratio = 0.2
    val_set_length = int(val_ratio * len(images))

    cur = np.random.randint(0, len(images))
    indexes = [cur]

    for _ in range(val_set_length):
        image = images[cur]
        mask = masks[cur]
        os.replace(path + "images/" + image, path + "val_images/" + image)
        os.replace(path + "masks/" + mask, path + "val_masks/" + mask)

        while (cur in indexes) and (os.path.isdir(images[cur]) is False):
            cur = np.random.randint(0, len(images))

        indexes.append(cur)

    for img in os.listdir(path + "train"):
        os.replace(path + "train/" + img, path + "train_images/" + img)


if __name__ == "__main__":
    #separate(path="C:/Users/user/PycharmProjects/Dastan/Unet/data/")
    #separate(path="C:/Users/user/PycharmProjects/Dastan/Unet/sandstone_data/decomposed/")
    pass
