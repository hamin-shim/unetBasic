import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from skimage.transform import resize


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET_3D(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNET_3D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)

        for feature in features:
            self.downs.append(DoubleConv3d(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv3d(feature * 2, feature))

        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            print(x.shape)
            skip_connections.append(x)
            x = self.pool(x)
        print('before bottleneck', x.shape)
        x = self.bottleneck(x)
        print('after bottleneck', x.shape)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            print("x", x.shape, "skip", skip_connection.shape)
            if x.shape != skip_connection.shape:
                skip_connection = resize(skip_connection, x.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((1, 1, 240, 240, 155))
    model = UNET_3D(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
