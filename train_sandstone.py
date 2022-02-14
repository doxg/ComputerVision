import pdb
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Model import Unet
from utils import read_args, get_loaders1, check_accuracy, save_predictions_as_imgs, save_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
args = read_args("config/configs.yaml")


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        # targets = targets.float().unsqueeze(1).to(DEVICE)
        targets = targets.float().to(DEVICE)

        with torch.cuda.amp.autocast():  ### Tensors may be any type, improve performance while maintaining accuracy
            preds = model(data)
            loss = loss_fn(preds.float(), targets.long())

        optimizer.zero_grad()
        scaler.scale(loss).backward()  #### To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor
                                        #### and invokes a backward pass on the scaled loss(es).
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item()) # update tqdm loop


def main():
    train_transforms = A.Compose([A.Rotate(limit=35, p=1.0),  #### Probability of applying transform = p
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.1),
                                  A.Normalize(mean=0.0,
                                             std=1.0,
                                             max_pixel_value=255.0),
                                  ToTensorV2()])

    eval_transforms = A.Compose([A.Normalize(mean=0.0,
                                             std=1.0,
                                             max_pixel_value=255.0),
                                 ToTensorV2()])

    model = Unet(in_channels=1, out_channels=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    train_loader, eval_loader = get_loaders1(args, train_transforms, eval_transforms)

    if args.LOAD_MODEL:
        print("Loading Checkpoint")
        checkpoint = torch.load("sandstone_checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])

    check_accuracy(eval_loader, model, DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        check_accuracy(eval_loader, model, DEVICE)

        save_predictions_as_imgs(
            eval_loader, model, folder="multisegmentation_results/", device=DEVICE
        )


if __name__ == "__main__":
    main()
