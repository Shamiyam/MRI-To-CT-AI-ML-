#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import json
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lpips
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandAffined,
    RandGaussianNoised, RandAdjustContrastd, ToTensord
)
from monai.networks.nets import UNet
from monai.utils import set_determinism


# --- Helper Functions for Data and Models ---

def make_transforms(train: bool):
    """Defines the MONAI transformations for training and validation."""
    load = [LoadImaged(keys=["src", "tgt"]), EnsureChannelFirstd(keys=["src", "tgt"])]
    aug = []
    if train:
        aug += [
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["src", "tgt"], prob=0.2,
                rotate_range=(math.radians(5),) * 3, scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "bilinear"), padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["src"], prob=0.1, std=0.01),
            RandAdjustContrastd(keys=["src"], prob=0.2, gamma=(0.9, 1.1)),
        ]
    to_tensor = [ToTensord(keys=["src", "tgt"])]
    return Compose(load + aug + to_tensor)

def load_split(csv_path: Path):
    """Loads a list of data dictionaries from a CSV file."""
    df = pd.read_csv(csv_path); data = [{"src": s, "tgt": t} for s, t in zip(df["src_nii"], df["tgt_nii"])]; return data

def make_generator():
    """Creates the 3D U-Net model which acts as the Generator."""
    return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="INSTANCE")

def make_discriminator(in_channels=2, base_ch=32, spectral_norm: bool = False):
    """3D PatchGAN discriminator with optional spectral norm."""
    def conv(cin, cout, k=4, s=2, p=1, use_sn=False, bias=True):
        m = nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=bias)
        return nn.utils.spectral_norm(m) if use_sn else m
    def block(cin, cout, stride, use_norm=True):
        layers = [conv(cin, cout, k=4, s=stride, p=1, use_sn=spectral_norm, bias=not use_norm)]
        if use_norm: layers.append(nn.InstanceNorm3d(cout, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True)); return layers
    return nn.Sequential(*block(in_channels, base_ch, stride=2, use_norm=False), *block(base_ch, base_ch * 2, stride=2), *block(base_ch * 2, base_ch * 4, stride=2), *block(base_ch * 4, base_ch * 8, stride=1), conv(base_ch * 8, 1, k=4, s=1, p=1, use_sn=spectral_norm, bias=True))

# --- Regularizers and helpers ---

class LPIPS_3D(nn.Module):
    """Wrapper to compute LPIPS loss slice-wise on 3D volumes."""
    def __init__(self, device):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
    def forward(self, pred_vol, tgt_vol):
        pred_scaled, tgt_scaled = pred_vol * 2.0 - 1.0, tgt_vol * 2.0 - 1.0
        depth = pred_scaled.shape[2]
        total_loss = 0.0
        for d in range(depth):
            pred_slice, tgt_slice = pred_scaled[:, :, d, :, :], tgt_scaled[:, :, d, :, :]
            total_loss += self.lpips_model(pred_slice.repeat(1,3,1,1), tgt_slice.repeat(1,3,1,1)).mean()
        return total_loss / depth

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items(): self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

# --- Core Training and Validation Functions ---

def train_one_epoch_gan(
    generator, discriminator, loader, optim_G, optim_D, scaler, device,
    l1_loss, gan_loss, lpips_loss, lambda_l1=100.0, lambda_lpips=0.1, 
    amp=True, use_hinge: bool = False, ema: EMA = None
):
    """Runs a single adversarial training epoch."""
    generator.train(); discriminator.train()
    meter = {"loss_D": 0.0, "loss_G": 0.0, "loss_G_gan": 0.0, "loss_G_l1": 0.0, "loss_G_lpips": 0.0}
    n_samples = 0
    for batch in loader:
        src, tgt = batch["src"].to(device, non_blocking=True), batch["tgt"].to(device, non_blocking=True)
        B = src.size(0); n_samples += B
        # --- Train Discriminator ---
        optim_D.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            fake_ct = generator(src).detach()
            pred_real = discriminator(torch.cat([src, tgt], dim=1))
            pred_fake = discriminator(torch.cat([src, fake_ct], dim=1))
            if use_hinge: 
                loss_D_real, loss_D_fake = torch.relu(1.0 - pred_real).mean(), torch.relu(1.0 + pred_fake).mean()
            else: 
                loss_D_real, loss_D_fake = gan_loss(pred_real, torch.ones_like(pred_real)), gan_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
        scaler.scale(loss_D).backward(); scaler.step(optim_D); scaler.update()

        # --- Train Generator ---
        optim_G.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            fake_ct_for_G = generator(src)
            pred_fake_for_G = discriminator(torch.cat([src, fake_ct_for_G], dim=1))
            if use_hinge: 
                loss_G_gan = -pred_fake_for_G.mean()
            else: 
                loss_G_gan = gan_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_l1 = l1_loss(fake_ct_for_G, tgt)
            loss_G_lpips = lpips_loss(fake_ct_for_G, tgt) if lpips_loss is not None else torch.zeros((), device=device)
            loss_G = loss_G_gan + (lambda_l1 * loss_G_l1) + (lambda_lpips * loss_G_lpips)
        scaler.scale(loss_G).backward(); scaler.step(optim_G); scaler.update()

        if ema is not None: ema.update(generator)
        meter["loss_D"] += loss_D.item() * B; meter["loss_G"] += loss_G.item() * B
        meter["loss_G_gan"] += loss_G_gan.item() * B; meter["loss_G_l1"] += loss_G_l1.item() * B; meter["loss_G_lpips"] += loss_G_lpips.item() * B
    for k in meter: meter[k] /= max(1, n_samples)
    return meter

@torch.no_grad()
def validate(model, loader, device):
    """Runs validation on the generator and returns metrics."""
    model.eval(); mae_sum, ssim_sum, n = 0.0, 0.0, 0
    from monai.losses import SSIMLoss # Re-import locally for safety in multiprocessing
    ssim_metric = SSIMLoss(spatial_dims=3, data_range=1.0); l1 = nn.L1Loss()
    for batch in loader:
        src, tgt = batch["src"].to(device, non_blocking=True), batch["tgt"].to(device, non_blocking=True)
        pred = model(src)
        mae_sum += l1(pred, tgt).item() * src.size(0)
        ssim_sum += (1.0 - ssim_metric(pred, tgt).item()) * src.size(0); n += src.size(0)
    return mae_sum / max(1, n), ssim_sum / max(1, n)

def save_validation_preview(batch, output, save_dir, epoch):
    """Saves a 2D slice preview."""
    src, tgt, pred = batch["src"][0, 0].cpu().numpy(), batch["tgt"][0, 0].cpu().numpy(), output[0, 0].cpu().detach().numpy()
    z = src.shape[2] // 2
    src_s, tgt_s, pred_s = src[:, :, z], tgt[:, :, z], pred[:, :, z]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axs, [src_s, tgt_s, pred_s], ["Source", "Target", "Prediction"]):
        ax.imshow(data.T, cmap="gray", origin="lower"); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"Epoch {epoch} Validation Sample"); plt.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(save_dir) / f"epoch_{epoch:03d}_sample.png", dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Train a 3D pix2pix GAN for MRI/CBCT → CT.")
    ap.add_argument("--splits", type=Path, required=True); ap.add_argument("--exp-name", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--save-dir", type=Path, default=Path("outputs/experiments"))
    ap.add_argument("--lambda-l1", type=float, default=100.0); ap.add_argument("--lrG", type=float, default=2e-4)
    ap.add_argument("--lrD", type=float, default=2e-4); ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--disc-base-ch", type=int, default=32); ap.add_argument("--spectral-norm", action="store_true")
    ap.add_argument("--hinge", action="store_true"); ap.add_argument("--ema", action="store_true")
    ap.add_argument("--lambda-lpips", type=float, default=0.0, help="Weight for LPIPS loss. Set > 0 to enable.")
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = (args.save_dir / args.exp_name).resolve()
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True); (exp_dir / "samples").mkdir(parents=True, exist_ok=True)

    train_ds = CacheDataset(data=load_split(args.splits / "train.csv"), transform=make_transforms(True), cache_rate=0.5, num_workers=args.num_workers)
    val_ds = CacheDataset(data=load_split(args.splits / "val.csv"), transform=make_transforms(False), cache_rate=0.5, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    generator = make_generator().to(device); discriminator = make_discriminator(in_channels=2, base_ch=args.disc_base_ch, spectral_norm=args.spectral_norm).to(device)
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=args.lrG, momentum=0.9)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=args.lrD, momentum=0.9)
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=10)
    scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=10)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    l1_loss, gan_loss = nn.L1Loss(), nn.BCEWithLogitsLoss()
    lpips_loss = LPIPS_3D(device) if args.lambda_lpips > 0 else None

    best_mae, start_epoch = float("inf"), 1
    if args.resume_from and args.resume_from.exists():
        ckpt = torch.load(args.resume_from, map_location=device)
        generator.load_state_dict(ckpt.get("generator", ckpt.get("model")))
        if "discriminator" in ckpt: discriminator.load_state_dict(ckpt["discriminator"])
        print("[INFO] Performing soft resume: loading model weights only.")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        print(f"[INFO] Resumed training from checkpoint. Starting at epoch {start_epoch}.")
    else:
        print("[INFO] Starting new GAN training run.")

    ema = EMA(generator) if args.ema else None

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch_gan(generator, discriminator, train_loader, optimizer_G, optimizer_D, scaler, device, l1_loss, gan_loss, lpips_loss, lambda_l1=args.lambda_l1, lambda_lpips=args.lambda_lpips, amp=args.amp, use_hinge=args.hinge, ema=ema)
        eval_model = make_generator().to(device)
        eval_model.load_state_dict(generator.state_dict())
        if ema is not None: ema.copy_to(eval_model)
        val_mae, val_ssim = validate(eval_model, val_loader, device)
        dt = time.time() - t0
        scheduler_G.step(val_mae); scheduler_D.step(val_mae)
        print(f"Epoch {epoch:03d}/{args.max_epochs} | D:{train_m['loss_D']:.4f} G:{train_m['loss_G']:.4f} "
              f"(GAN:{train_m['loss_G_gan']:.4f} L1:{train_m['loss_G_l1']:.4f} LPIPS:{train_m['loss_G_lpips']:.4f}) | "
              f"val_MAE={val_mae:.4f} | val_SSIM={val_ssim:.4f} | {dt:.1f}s")

        if val_mae < best_mae:
            best_mae = val_mae
            best_ckpt_path = exp_dir / "checkpoints" / "best.pt"
            torch.save({"epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_G": optimizer_G.state_dict(), "optimizer_D": optimizer_D.state_dict(), "best_mae": best_mae, "scaler": scaler.state_dict()}, best_ckpt_path)
            print(f"[INFO] Saved best GAN checkpoint: {best_ckpt_path}")
            val_batch = next(iter(val_loader))
            save_validation_preview(val_batch, eval_model(val_batch["src"].to(device)), exp_dir / "samples", epoch)
    
        last_ckpt_path = exp_dir / "checkpoints" / "last.pt"
        torch.save({"epoch": epoch, "generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_G": optimizer_G.state_dict(), "optimizer_D": optimizer_D.state_dict(), "best_mae": best_mae, "scaler": scaler.state_dict()}, last_ckpt_path)

    print(f"[INFO] Training complete. Best val MAE: {best_mae:.4f} -> {exp_dir}")

if __name__ == "__main__":
    main()

