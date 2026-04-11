from pathlib import Path
import sys
import torch.nn as nn
import torch.nn.functional as F
import importlib
import os

#import models
from models.MR_LKV_image import MR_LKV
from models.UNet import UNet
from models.ReplkNet import RepLKNet
from ExternalRepo.SwinIR.models.network_swinir import SwinIR

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

# Add src
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "ExternalRepo", "Restormer"))
from ExternalRepo.Restormer.basicsr.models.archs.restormer_arch import Restormer as RestormerNet 




class RepLKNetReg(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = RepLKNet(
            in_channels=1,
            out_channels=1
        )

    def forward(self, x):
        return self.net(x)


class SwinIRWrapper(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = SwinIR(
            img_size=128,
            window_size=8,
            in_chans=1,
            out_chans=1,
            depths=[2,2,2,2],
            embed_dim=48,
            num_heads=[2,2,2,2],
            upsampler="none"
        )

    def forward(self,x):
        B, C, H, W = x.shape
        x_resized = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
        return self.net(x)
class RestormerWrapper(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48, #48
                 num_blocks=[2, 2, 2, 3], 
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.0, 
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):
                 super().__init__()
                 self.net = RestormerNet(
                    inp_channels=inp_channels,
                    out_channels=out_channels,
                    dim=dim,
                    num_blocks=num_blocks,
                    num_refinement_blocks=num_refinement_blocks,
                    heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                    dual_pixel_task=dual_pixel_task
                    )
    def forward(self, x):
        return self.net(x)

def build_model(model_name):

    if model_name == "mr_lkv":

        return MR_LKV(
            in_channels=1,
            base_channels=32,
            depths=[2,2,3,2],
            kernels=[35,55,75,95],
            norm_type="batch"
        )

    elif model_name == "unet":

        return UNet(
            in_channels=1,
            base_channels=32,
            levels=4,
            norm_type=None
        )

    elif model_name == "replknet":
        return RepLKNet(
            in_channels=1,
            out_channels=1
        )

    elif model_name == "swinir":

        return SwinIRWrapper()

    elif model_name == "restormer":

        return RestormerWrapper()

    else:

        raise ValueError("Unknown model")