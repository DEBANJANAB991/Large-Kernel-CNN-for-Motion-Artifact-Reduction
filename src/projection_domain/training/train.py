#!/usr/bin/env python3
"""
Module: Projection-Domain Training for CT Artifact Reduction

This code trains deep learning models (MR-LKV, UNet, RepLKNet, SwinIR, Restormer)
on reconstructed CT images affected by motion artifacts.

Pipeline:
1. Load 2D clean and artifact sinogram pairs from .npy files
2. Train models to map artifact → clean images
3. Evaluate using PSNR and SSIM
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import random

#model imports
from models.MR_LKV_projection import MR_LKV      # MR-LKV implementation
from models.UNet import UNet                     # U-Net baseline
from models.ReplkNet import RepLKNet             # Original RepLKNet backbone
from ExternalRepo.SwinIR.models.network_swinir import SwinIR #SwinIR import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
# Add src
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "ExternalRepo", "Restormer"))
from ExternalRepo.Restormer.basicsr.models.archs.restormer_arch import Restormer as RestormerNet #Restormer import

# Config import
from config.config import (
    CLEAN_SINOGRAM_2D,
    ARTIFACT_ROOT_2D,
    CKPT_DIR,
    BATCH_SIZE,
    LR,
    EPOCHS,
    SAVE_INTERVAL,
)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ---------------- Metrics ---------------- #
def psnr(pred, target, max_val: float = 1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    return 10 * torch.log10(max_val**2 / (mse + 1e-12))
   
def gaussian(window_size: int, sigma: float):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def create_window(window_size: int, channel: int):
    _1d = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, K=(0.01, 0.03), L: float = 1.0):
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ---------------- Data splitting ---------------- #
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 1.0 - TRAIN_FRAC - VAL_FRAC

def parse_patient_from_filename(fname: str):
    """
    Parses patient ID from filename format:
      'CQ500CTXXX CQ500CTXXX_viewXXX.npy'
    Returns: patient_id (e.g. 'CQ500CTXXX')
    """
   
    base = Path(fname).stem
    parts = base.split(" ")
    if len(parts) >= 1:
        return parts[0]
    return base


def adjust_replk_input(x, y):
    _, _, H, W = x.shape

    H_new = (H // 4) * 4
    W_new = (W // 4) * 4

    return x[:, :, :H_new, :W_new], y[:, :, :H_new, :W_new]

class Sinogram2DDataset(Dataset):
    """
    Dataset that groups files by patient and samples up to max_views_per_patient views per patient.
    Each item returns (artifact_tensor, clean_tensor) with shape (1,H,W) normalized to 0..1.

    If patch_size is provided (>0) -> random crop patch_size x patch_size per sample.
    Otherwise returns full-size sinogram (no cropping).
    """
    
    def __init__(self, clean_root: Path, art_root: Path, max_views_per_patient: int = 300, patch_size: int = 0):
        self.clean_root = Path(clean_root)
        self.art_root = Path(art_root)
        if not self.clean_root.exists():
            raise RuntimeError(f"Clean root does not exist: {self.clean_root}")
        if not self.art_root.exists():
            raise RuntimeError(f"Art root does not exist: {self.art_root}")

        # collect all clean files
        clean_files = sorted([p for p in self.clean_root.rglob("*.npy")])
        if not clean_files:
            raise RuntimeError(f"No .npy files under {self.clean_root}")

        # group by patient
        by_patient = {}
        for p in clean_files:
            patient = parse_patient_from_filename(p.name)
            by_patient.setdefault(patient, []).append(p)

        # per patient, sample up to max_views_per_patient
        selected_pairs = []
        missing_art = []
        for patient, files in by_patient.items():
            files = sorted(files)
            if max_views_per_patient is not None and len(files) > max_views_per_patient:
                files = random.sample(files, max_views_per_patient)
                files = sorted(files)  # keep deterministic-ish order after sampling
            for cpath in files:
                art_path = self.art_root / cpath.name  # same filename expected in artifact root
                if not art_path.exists():
                    matches = list(self.art_root.rglob(cpath.name))
                    if matches:
                        art_path = matches[0]
                    else:
                        missing_art.append(str(art_path))
                        continue
                selected_pairs.append((cpath, art_path))

        if missing_art:
            print("Warning: Missing artifact files for some clean sinograms. Missing example(s):")
            for m in missing_art[:5]:
                print("  ", m)
            print(f"Total missing artifacted files: {len(missing_art)}")

        if not selected_pairs:
            raise RuntimeError("No paired clean/artifact files found after filtering.")

        # final list
        self.pairs = selected_pairs

        self.patch_size = int(patch_size) if patch_size is not None else 0
        print(f"Dataset: {len(self.pairs)} paired 2D sinograms loaded from {self.clean_root} / {self.art_root}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
       
        cpath, apath = self.pairs[idx]
       
        clean = np.load(cpath, mmap_mode='r').astype(np.float32)
        art   = np.load(apath, mmap_mode='r').astype(np.float32)
        if clean.ndim == 3 and clean.shape[0] != 1:
            # central 2D slice
            zc = clean.shape[0] // 2
            clean = clean[zc]
        if art.ndim == 3 and art.shape[0] != 1:
            zc = art.shape[0] // 2
            art = art[zc]
        a_min = art.min()
        a_max = art.max()
        #Normalize using artifact slice statistics to preserve relative intensity
        #differences between corrupted and clean images     
        denom = a_max - a_min + 1e-12
        art_norm   = (art   - a_min) / denom
        clean_norm = (clean - a_min) / denom
        H, W = clean_norm.shape    

        clean_t = torch.from_numpy(clean_norm).unsqueeze(0).float()
        art_t   = torch.from_numpy(art_norm).unsqueeze(0).float()
        return art_t, clean_t


# ---------------- SwinIRWrapper---------------- #
class SwinIRWrapper(nn.Module):
    def __init__(self, img_size=512, window_size=8, in_chans=1, out_chans=1,*, use_checkpoint: bool = False, **kwargs):
        super().__init__()
        self.net = SwinIR(
            img_size=img_size,
            window_size=window_size,
            in_chans=in_chans,
            out_chans=out_chans,
            img_range=1.0,
            upsampler='none',
            use_checkpoint=use_checkpoint,
            **kwargs
        )
    def forward(self, x):
        B, C, H, W = x.shape
        H_new = (H // 8) * 8
        W_new = (W // 8) * 8
        x = x[:, :, :H_new, :W_new]
        return self.net(x)
# ---------------- RestormerWrapper ---------------- #
class RestormerWrapper(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48,
                 num_blocks=[2,2,2,3],
                 num_refinement_blocks=2,
                 heads=[1,2,4,8],
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
        B, C, H, W = x.shape
        H_new = (H // 8) * 8
        W_new = (W // 8) * 8
        x = x[:, :, :H_new, :W_new]
        return self.net(x)


# ---------------- Args ---------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Train artifact-reduction models (2D sinogram views)")
    p.add_argument("--model", choices=["mr_lkv", "unet", "replk","swinir","restormer"], default="mr_lkv")
    p.add_argument("--clean-root",   type=Path, default=Path(CLEAN_SINOGRAM_2D))
    p.add_argument("--art-root",     type=Path, default=Path(ARTIFACT_ROOT_2D)) 
    p.add_argument("--ckpt-dir",     type=Path, default=Path(CKPT_DIR))
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--save-interval",type=int, default=SAVE_INTERVAL)
    p.add_argument("--base-ch",      type=int, default=32, help="Base channels for MR_LKV")
    p.add_argument("--norm", choices=["batch","instance","group", "layer","none"], default="batch")
    p.add_argument("--no-decoder", action="store_true", help="Disable decoder in MR_LKV")
    p.add_argument("--unet-base", type=int, default=32)
    # replk args (kept)
    p.add_argument("--replk-kernels",   nargs=4, type=int, default=[21, 19, 17, 9])
    p.add_argument("--replk-layers",    nargs=4, type=int, default=[2,2,3,2])
    p.add_argument("--replk-channels",  nargs=4, type=int, default=[32, 64, 128, 256])
    p.add_argument("--replk-small",     type=int, default=5)
    p.add_argument("--replk-drop-path", type=float, default=0.0)
    # dataset sampling
    p.add_argument("--max-views-per-patient", type=int, default=100,
                   help="Maximum number of 2D views to sample per patient (random sample if more exist).")
    p.add_argument("--patch", type=int, default=0,
                   help="If >0, random crop patch of size patch x patch. Set 0 to disable cropping and use full images.")
    p.add_argument("--resume", type=Path, default=None,
               help="Path to checkpoint to resume training")

    return p.parse_args()

# ---------------- Checkpoint folder mapping ---------------- #
_FOLDER_MAP = {
    "mr_lkv": "mr_lkv",
    "unet": "unet",
    "replk": "replknet",
    "swinir": "swinir",
    "restormer": "restormer",

}

def _model_dir(ckpt_root: Path, model_name: str) -> Path:
    key = model_name.lower()
    folder = _FOLDER_MAP.get(key, key)
    return Path(ckpt_root) / folder


# ---------------- Train & Eval ---------------- #
def main():
    args = parse_args()
  

    model_ckpt_dir = _model_dir(args.ckpt_dir, args.model)
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Checkpoints will be saved to: {model_ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    dataset = Sinogram2DDataset(
    args.clean_root,
    args.art_root,
    max_views_per_patient=args.max_views_per_patient,
    patch_size=args.patch)
    
    pairs = dataset.pairs
    patients = list(set(parse_patient_from_filename(p[0].name) for p in pairs))
    random.shuffle(patients)
    
    n_train = int(TRAIN_FRAC * len(patients))
    n_val   = int(VAL_FRAC * len(patients))
    
    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train+n_val])
    test_patients  = set(patients[n_train+n_val:])

    train_ds = torch.utils.data.Subset(dataset, [i for i, (cpath, _) in enumerate(pairs) if parse_patient_from_filename(cpath.name) in train_patients])

    val_ds = torch.utils.data.Subset(dataset, [i for i, (cpath, _) in enumerate(pairs) if parse_patient_from_filename(cpath.name) in val_patients])

    test_ds = torch.utils.data.Subset(dataset, [i for i, (cpath, _) in enumerate(pairs) if parse_patient_from_filename(cpath.name) in test_patients])
    

   
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # build model
    if args.model == "unet":
        model = UNet(in_channels=1, base_channels=32, levels=4, norm_type=None, dropout_bottleneck=0.1, final_activation=None).to(device)
    elif args.model == "mr_lkv":
        model = MR_LKV(
            in_channels=1,
            base_channels=args.base_ch,
            depths=[2,2,3,2],
            kernels=[35,55,75,95],
            norm_type=args.norm,
            use_decoder=(not args.no_decoder),
            final_activation=None
        ).to(device)
    elif args.model == "replk":
        model = RepLKNet(in_channels=1, out_channels=1).to(device)
    elif args.model == "swinir": 
        model = SwinIRWrapper(img_size=128, window_size=8, in_chans=1, out_chans=1, depths=[2,2,2,2], embed_dim=48, num_heads=[2,2,2,2], use_checkpoint=False).to(device)
    elif args.model == "restormer":
        model = RestormerWrapper(inp_channels=1, out_channels=1, dim=48, num_blocks=[2,2,2,3], num_refinement_blocks=2, heads=[1,2,4,8], ffn_expansion_factor=2.0).to(device)
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    if args.model == "replk":
        lr = LR * 0.1  
    elif args.model in ["restormer", "swinir"]:
        lr = LR * 0.2   
    else:
        lr = LR * 0.25        

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    criterion = nn.L1Loss() 
   
   
  
    torch.cuda.empty_cache()
    best_val = float('inf')
    start_epoch = 1
    epochs_no_imp = 0
    EARLY_STOP_PATIENCE = 10
    # ---------------- RESUME CHECKPOINT ----------------
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)

        print(f"Resumed from epoch {checkpoint['epoch']} | best_val = {best_val:.6f}")

    # logging
    results_dir = Path(PROJECT_ROOT) / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f"{args.model}_training_log.csv"
    if args.resume is None or not log_path.exists():
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_psnr,val_ssim\n")

    train_losses, val_losses, psnr_scores, ssim_scores = [], [], [], []
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):

        model.train()
        running_train = 0.0
        for art, clean in train_loader:
            if args.model == "replk":
                art, clean = adjust_replk_input(art, clean)
            art, clean = art.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(art)
            if pred.shape[-2:] != clean.shape[-2:]:
                pred = F.interpolate(pred, size=clean.shape[-2:], mode='bilinear', align_corners=False)

    
            loss = criterion(pred, clean)
            

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            running_train += loss.item() * art.size(0)
        train_loss = running_train / max(1, len(train_ds))

        # Evaluate model on validation set
        model.eval()
        running_val = running_psnr = running_ssim = 0.0
        with torch.no_grad():
            for art, clean in val_loader:
                if args.model == "replk":
                    art, clean = adjust_replk_input(art, clean)
                art, clean = art.to(device), clean.to(device)
                pred = model(art)
                
                if pred.shape[-2:] != clean.shape[-2:]:
                    pred = F.interpolate(pred, size=clean.shape[-2:], mode='bilinear', align_corners=False)
                running_val += criterion(pred, clean).item() * art.size(0)
                running_psnr += psnr(pred, clean).item() * art.size(0)
                running_ssim += ssim(pred, clean).item() * art.size(0)
        val_loss = running_val / max(1, len(val_ds))
        val_psnr = running_psnr / max(1, len(val_ds))
        val_ssim = running_ssim / max(1, len(val_ds))

        print(f"Epoch {epoch}/{args.epochs} — Train: {train_loss:.6f}, Val: {val_loss:.6f} | PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f}\n")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_scores.append(val_psnr)
        ssim_scores.append(val_ssim)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_imp = 0
            torch.save({ "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_val": best_val }, model_ckpt_dir / "best_model.pth")

        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        if epoch % args.save_interval == 0:
            ckpt = model_ckpt_dir / f"epoch{epoch}.pth"
            

            torch.save({"epoch": epoch,"model": model.state_dict(),"optimizer": optimizer.state_dict(),"scheduler": scheduler.state_dict(),"best_val": best_val}, ckpt)


    # plot training curves
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(psnr_scores, label="Val PSNR (dB)", marker='s')
    plt.plot(ssim_scores, label="Val SSIM", marker='s')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

   
    plot_path = results_dir / f"{args.model}_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves to {plot_path}")

    #Final evaluation on unseen test set for unbiased performance estimation
    model.eval()
    running_test = running_psnr = running_ssim = 0.0
    with torch.no_grad():
        for batch_idx, (art, clean) in enumerate(test_loader):
            if args.model == "replk":
                art, clean = adjust_replk_input(art, clean)
            art, clean = art.to(device), clean.to(device)
            pred = model(art)

            if pred.shape[-2:] != clean.shape[-2:]:
                pred = F.interpolate(pred, size=clean.shape[-2:], mode='bilinear', align_corners=False)
            running_test += criterion(pred, clean).item() * art.size(0)
            running_psnr += psnr(pred, clean).item() * art.size(0)
            running_ssim += ssim(pred, clean).item() * art.size(0)
    test_loss = running_test / max(1, len(test_ds))
    test_psnr = running_psnr / max(1, len(test_ds))
    test_ssim = running_ssim / max(1, len(test_ds))
    print(f"Final Test — Loss: {test_loss:.6f}, PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")

if __name__ == "__main__":
    main()