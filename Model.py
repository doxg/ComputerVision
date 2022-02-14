import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolution, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))


    def forward(self, x):

        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))



class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ### Downsampling ###

        for feature in features:
            self.downs.append(DoubleConvolution(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        ### Upsampling ###

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=(2, 2), stride=2)) ### Upsampling [OddIndexes]
            self.ups.append(DoubleConvolution(feature*2, feature)) #### Convolution [EvenIndexes]

        self.bottleneck = DoubleConvolution(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=(1, 1))

    def forward(self, x):
        skip_conns = []
        for down in self.downs:
            x = down(x)
            skip_conns.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_conns = skip_conns[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_conns[idx//2]

            if x.shape != skip_conn.shape:
                x = tf.resize(x, size=skip_conn.shape[2:]) ### In paper they use crop

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        out = self.final_conv(x)

        return out


def test():
    x = torch.randn((3, 1, 160, 160))
    model = Unet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape, x.shape)


if __name__ == "__main__":
    test()
