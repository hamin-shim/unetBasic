import torch.nn as nn
import torch
import math


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv3d(out_channels,
                                             out_channels, 3, padding=1),
                                   nn.GroupNorm(1, out_channels),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, x):
        x = self.dconv(x)
        return x


class down_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down = nn.Sequential(nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
                                  double_conv(in_channels, out_channels))

    def forward(self, x):
        x = self.down(x)
        return x


class UNETPP(nn.Module):
    def __init__(self, model_info, init_weights=True):
        super().__init__()

        self.up_op = nn.Upsample(scale_factor=(
            1, 2, 2), mode='trilinear', align_corners=True)

        self.stem = double_conv(
            model_info['in_channel'], model_info['encoder'][0][0])

        encoder = []
        for i, layer_info in enumerate(model_info["encoder"]):
            in_channels = layer_info[0]
            out_channels = layer_info[1]
            encoder.append(down_op(in_channels, out_channels))

        self.encoder = nn.ModuleList([*encoder])

        nest_conv0_x = []
        for i in range(4):
            nest_conv0_x.append(double_conv((32 + 16 * (i + 1)), 16))

        self.nest_conv0 = nn.ModuleList([*nest_conv0_x])

        nest_conv1_x = []
        for i in range(3):
            nest_conv1_x.append(double_conv((64 + 32 * (i + 1)), 32))

        self.nest_conv1 = nn.ModuleList([*nest_conv1_x])

        nest_conv2_x = []
        for i in range(2):
            nest_conv2_x.append(double_conv((128 + 64 * (i + 1)), 64))

        self.nest_conv2 = nn.ModuleList([*nest_conv2_x])

        nest_conv3_x = []
        nest_conv3_x.append(double_conv((256 + 128), 128))

        self.nest_conv3 = nn.ModuleList([*nest_conv3_x])

        self.last = nn.Conv3d(16, 3, 1)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        print(x.shape)
        # x = self.conv1x1(x)
        x0_0 = self.stem(x)
        print(x0_0.shape)
        x1_0 = self.encoder[0](x0_0)
        print(x1_0.shape)
        x2_0 = self.encoder[1](x1_0)
        print(x2_0.shape)
        x3_0 = self.encoder[2](x2_0)
        print(x3_0.shape)
        x4_0 = self.encoder[3](x3_0)
        print(x4_0.shape)

        x0_1 = self.nest_conv0[0](torch.concat(
            [x0_0, self.up_op(x1_0)], dim=1))
        print(x0_1.shape)
        x1_1 = self.nest_conv1[0](torch.concat(
            [x1_0, self.up_op(x2_0)], dim=1))
        print(x1_1.shape)
        x2_1 = self.nest_conv2[0](torch.concat(
            [x2_0, self.up_op(x3_0)], dim=1))
        print(x2_1.shape)
        x3_1 = self.nest_conv3[0](torch.concat(
            [x3_0, self.up_op(x4_0)], dim=1))
        print(x3_1.shape)

        x0_2 = self.nest_conv0[1](torch.concat(
            [x0_0, x0_1, self.up_op(x1_1)], dim=1))
        print(x0_2.shape)
        x1_2 = self.nest_conv1[1](torch.concat(
            [x1_0, x1_1, self.up_op(x2_1)], dim=1))
        print(x1_2.shape)
        x2_2 = self.nest_conv2[1](torch.concat(
            [x2_0, x2_1, self.up_op(x3_1)], dim=1))
        print(x2_2.shape)

        x0_3 = self.nest_conv0[2](torch.concat(
            [x0_0, x0_1, x0_2, self.up_op(x1_2)], dim=1))
        print(x0_3.shape)
        x1_3 = self.nest_conv1[2](torch.concat(
            [x1_0, x1_1, x1_2, self.up_op(x2_2)], dim=1))
        print(x1_3.shape)
        x0_4 = self.nest_conv0[3](torch.concat(
            [x0_0, x0_1, x0_2, x0_3, self.up_op(x1_3)], dim=1))
        print(x0_4.shape)
        x = self.last(x0_4)
        print(x.shape)
        return torch.sigmoid(x)


def unet_pp():
    model_info = {
        "encoder": [
            [32, 64],
            [64, 128],
            [128, 256],
            [256, 512],
        ],
    }
    return UNETPP(model_info)
