import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Normalization
# =========================================================
def norm_layer(channels):
    return nn.GroupNorm(1, channels)


def match_size(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

# =========================================================
# ConvFFN Block 
# =========================================================
class ConvFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden_dim = dim * expansion

        self.net = nn.Sequential(
            norm_layer(dim),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)


# =========================================================
# RepLK Block 
# =========================================================
class RepLKBlock(nn.Module):
    def __init__(self, dim, kernel_size=21, small_kernel=5):
        super().__init__()

        padding = kernel_size // 2
        small_padding = small_kernel // 2

        # Large kernel depthwise
        self.large_dw = nn.Conv2d(
            dim, dim, kernel_size,
            padding=padding,
            groups=dim,
            bias=False
        )

        # Small kernel (optimization helper)
        self.small_dw = nn.Conv2d(
            dim, dim, small_kernel,
            padding=small_padding,
            groups=dim,
            bias=False
        )

        self.norm = norm_layer(dim)
        self.pw1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.large_dw(x) + self.small_dw(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.pw1(out)

        return out + identity


# =========================================================
# RepLK Stage (Block + FFN)
# =========================================================
class RepLKStage(nn.Module):
    def __init__(self, dim, depth, kernel_size):
        super().__init__()

        layers = []
        for _ in range(depth):
            layers.append(RepLKBlock(dim, kernel_size))
            layers.append(ConvFFN(dim))

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


# =========================================================
# Upsample Block 
# =========================================================
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


# =========================================================
# FINAL MODEL (Reconstruction version)
# =========================================================
class RepLKNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Reduced dimensions
        dims = [32, 64, 128, 256]
        depths = [2, 2, 3, 2]
        kernels = [21, 19, 17, 9]

        # ---------------- Encoder ---------------- #
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1),
            norm_layer(dims[0]),
            nn.GELU()
        )

        self.stage1 = RepLKStage(dims[0], depths[0], kernels[0])
        self.down1 = nn.Conv2d(dims[0], dims[1], 2, stride=2)

        self.stage2 = RepLKStage(dims[1], depths[1], kernels[1])
        self.down2 = nn.Conv2d(dims[1], dims[2], 2, stride=2)

        self.stage3 = RepLKStage(dims[2], depths[2], kernels[2])
        self.down3 = nn.Conv2d(dims[2], dims[3], 2, stride=2)

        self.stage4 = RepLKStage(dims[3], depths[3], kernels[3])

        # ---------------- Decoder (WITH SKIPS) ---------------- #
        self.up3 = UpBlock(dims[3], dims[2])
        self.up2 = UpBlock(dims[2], dims[1])
        self.up1 = UpBlock(dims[1], dims[0])

        self.final = nn.Conv2d(dims[0], out_channels, 1)

    def forward(self, x):

        # -------- Encoder -------- #
        x1 = self.stage1(self.stem(x))   # [B, 32, H, W]
        x2 = self.stage2(self.down1(x1)) # [B, 64, H/2, W/2]
        x3 = self.stage3(self.down2(x2)) # [B,128,H/4,W/4]
        x4 = self.stage4(self.down3(x3)) # [B,256,H/8,W/8]

        # -------- Decoder with SKIP CONNECTIONS -------- #
        d3 = self.up3(x4)
        d3 = match_size(d3, x3)
        d3 = d3 + x3

        d2 = self.up2(d3)
        d2 = match_size(d2, x2)
        d2 = d2 + x2

        d1 = self.up1(d2)
        d1 = match_size(d1, x1)
        d1 = d1 + x1

        out = self.final(d1)

        return out