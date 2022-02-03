import pdb
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from main import Unet
from utils import read_args, get_loaders, evaluate_biseg, save_predictions_as_imgs, save_checkpoint, load_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
args = read_args("config/hyper_params.yaml")


def train_fn(loader, model, optimizer, loss_fn, scaler, load=False):
    if load:
        load_model(model,optimizer, "weights/binarysegmentation.pth.tar")

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
    train_transforms = A.Compose([A.Resize(height=args.IMAGE_HEIGHT, width=args.IMAGE_WIDTH),
                                  A.Rotate(limit=35, p=1.0),  #### Probability of applying transform = p
                                  A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.1),
                                  A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[1.0, 1.0, 1.0],
                                             max_pixel_value=255.0),
                                  ToTensorV2()])

    eval_transforms = A.Compose([A.Resize(height=args.IMAGE_HEIGHT, width=args.IMAGE_WIDTH),
                                 A.Normalize(mean=[0.0, 0.0, 0.0],
                                             std=[1.0, 1.0, 1.0],
                                             max_pixel_value=255.0),
                                 ToTensorV2()])

    model = Unet(in_channels=3, out_channels=2).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)

    train_loader, eval_loader = get_loaders(args, train_transforms, eval_transforms)


    dice_score = evaluate_biseg(eval_loader, model, DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, args.LOAD_MODEL)

        if dice_score < evaluate_biseg(eval_loader, model, DEVICE):

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, filename="weights/binarysegmentation.pth.tar")
            args.LOAD_MODEL = True
        else:
            args.LOAD_MODEL = False

        save_predictions_as_imgs(
            eval_loader, model, folder="results/binary_segmentation", device=DEVICE
        )


if __name__ == "__main__":
    main()

