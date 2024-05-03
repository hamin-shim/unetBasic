import torch.nn as nn
from einops import rearrange
from vitUnet_utils import *


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        # assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.pd, self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, dim, kernel_size)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv2 = conv_nxn_bn(dim, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # print("y", y.shape)
        # Local representations
        x = self.conv1(x)
        # print("conv1", x.shape)
        # Global representations
        _, _, d, h, w = x.shape
        x = rearrange(x, 'b c (d pd) (h ph) (w pw) -> b (pd ph pw) (d h w) c',
                      pd=self.pd, ph=self.ph, pw=self.pw)
        # print("after rearrange", x.shape)
        x = self.transformer(x)
        # print("after transformer", x.shape)
        x = rearrange(x, 'b (pd ph pw) (d h w) c -> b c (d pd) (h ph) (w pw)',
                      d=d//self.pd, h=h//self.ph, w=w//self.pw, pd=self.pd, ph=self.ph, pw=self.pw)
        # print("after rearrange 2", x.shape)

        # Fusion
        x = self.conv2(x)
        # print("conv2", x.shape)
        return x


class MUnetViT(nn.Module):
    def __init__(self, image_size, dims, channels, n_classes, expansion=4, kernel_size=3, patch_size=(2, 4, 4)):
        super().__init__()
        id, ih, iw = image_size
        pd, ph, pw = patch_size
        assert id % pd == 0 and ih % ph == 0 and iw % pw == 0

        L = [2, 4]

        self.conv1 = conv_nxn_bn(3, channels[0], stride=1)

        self.mv2 = nn.ModuleList([
            MV2Block(channels[0], channels[1], (1, 2, 2), expansion),
            MV2Block(channels[1], channels[2], (1, 2, 2), expansion),
            MV2Block(channels[2], channels[3], (1, 2, 2), expansion),
            MV2Block(channels[3], channels[4], (1, 2, 2), expansion)
        ])

        self.mvit = nn.ModuleList([
            MobileViTBlock(
                dims[0], L[0], channels[2], kernel_size, patch_size, int(dims[0]*2)),
            MobileViTBlock(
                dims[1], L[1], channels[3], kernel_size, patch_size, int(dims[1]*4))
        ])

        self.decs = nn.ModuleList([
            Up(2*channels[4], channels[2]),
            Up(channels[3], channels[1]),
            Up(channels[2], channels[0]),
        ])
        self.out = Out(channels[0], n_classes)

    def forward(self, x):
        print(f"Initial ({x.shape})")
        x_conv1 = self.conv1(x)
        print(f"conv1 ({x_conv1.shape})")

        x_mv0 = self.mv2[0](x_conv1)
        print(f"mv0 ({x_mv0.shape})")

        x_mv1 = self.mv2[1](x_mv0)
        print(f"mv1 ({x_mv1.shape})")
        x_mvit0 = self.mvit[0](x_mv1)
        print(f"mvit0 ({x_mvit0.shape})")

        x_mv2 = self.mv2[2](x_mvit0)
        print(f"mv2 ({x_mv2.shape})")
        x_mvit1 = self.mvit[1](x_mv2)
        print(f"mvit0 ({x_mvit0.shape})")

        x_mv3 = self.mv2[3](x_mvit1)
        print(f"mv3 ({x_mv3.shape})")

        mask = self.decs[0](x_mv3, x_mv2)
        print(f"mask 1 ({mask.shape})")
        mask = self.decs[1](mask, x_mv1)
        print(f"mask 2 ({mask.shape})")
        mask = self.decs[2](mask, x_mv0)
        print(f"mask 3 ({mask.shape})")
        mask = self.out(mask)
        return mask
