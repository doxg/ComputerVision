import pdb
import torch.nn.functional as F
import yaml
from datasets.dataset import MyDataset
from datasets.dataset_sandstone import SandStoneDataset
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def read_args(path):
    with open(path, 'r', encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.Loader)
        f.close()

    return args


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def get_loaders1(args, train_transforms, eval_transforms):
    train_ds = SandStoneDataset(image_dir="./sandstone_data/decomposed/train_images", mask_dir="./sandstone_data/decomposed/train_masks", transform=train_transforms)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.BATCH_SIZE, shuffle=True, pin_memory=args.PIN_MEMORY, num_workers=args.NUM_WORKERS) #### speed up the host to device transfer by enabling pin_memory.

    eval_ds = SandStoneDataset(image_dir="./sandstone_data/decomposed/val_images", mask_dir="./sandstone_data/decomposed/val_masks", transform=eval_transforms)
    eval_loader = DataLoader(dataset=eval_ds, batch_size=args.BATCH_SIZE, pin_memory=args.PIN_MEMORY, shuffle=False, num_workers=args.NUM_WORKERS)

    return train_loader, eval_loader


def get_loaders(args, train_transforms, eval_transforms):
    train_ds = MyDataset(image_dir=args.TRAIN_IMG_DIR, mask_dir=args.TRAIN_MASK_DIR, transform=train_transforms)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.BATCH_SIZE, shuffle=True, pin_memory=args.PIN_MEMORY, num_workers=args.NUM_WORKERS) #### speed up the host to device transfer by enabling pin_memory.

    eval_ds = MyDataset(image_dir=args.VAL_IMG_DIR, mask_dir=args.VAL_MASK_DIR, transform=eval_transforms)
    eval_loader = DataLoader(dataset=eval_ds, batch_size=args.BATCH_SIZE, pin_memory=args.PIN_MEMORY, shuffle=False, num_workers=args.NUM_WORKERS)

    return train_loader, eval_loader


def check_accuracy(val_loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            #preds = torch.sigmoid(model(x))
            #pdb.set_trace()
            preds = model(x)
            preds = preds.data.max(1, keepdim=True)[1]
            num_correct += preds.eq(y.data.view_as(preds)).cpu().sum()
            #num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(val_loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            # preds = torch.sigmoid(model(x))
            preds = model(x)
            preds = preds.data.max(1, keepdim=True)[1]
            #preds = (preds > 0.5).float()
        pdb.set_trace()
        # plot_compare(preds[0],y[0])
        save_image(preds.float(), f"{folder}/pred_{idx}.png")
        save_image(y.unsqueeze(1).float(), f"{folder}/{idx}.png")
        # save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()



def plot_compare(predicted, y):
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.title('Mask')
    plt.imshow(y.cpu(), cmap='jet')
    plt.subplot(222)
    plt.title('Prediction on test image')
    plt.imshow(predicted.cpu().squeeze(0), cmap='jet')
    plt.show()









