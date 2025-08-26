#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandFlipd, RandAffined,
    RandGaussianNoised, RandAdjustContrastd, ToTensord
)
from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from monai.utils import set_determinism


def make_transforms(train: bool):
    """Defines the MONAI transformations for training and validation."""
    load = [
        LoadImaged(keys=["src", "tgt"]),
        EnsureChannelFirstd(keys=["src", "tgt"]),
    ]
    aug = []
    if train:
        aug += [
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["src", "tgt"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["src", "tgt"],
                prob=0.2,
                rotate_range=(math.radians(5),) * 3,
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "bilinear"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["src"], prob=0.1, std=0.01),
            RandAdjustContrastd(keys=["src"], prob=0.2, gamma=(0.9, 1.1)),
        ]
    to_tensor = [ToTensord(keys=["src", "tgt"])]
    return Compose(load + aug + to_tensor)


def load_split(csv_path: Path):
    """Loads a list of data dictionaries from a CSV file."""
    df = pd.read_csv(csv_path)
    if not {"src_nii", "tgt_nii"}.issubset(df.columns):
        raise ValueError("Split CSV must contain columns: src_nii, tgt_nii")
    data = [{"src": s, "tgt": t} for s, t in zip(df["src_nii"], df["tgt_nii"])]
    return data


def make_model():
    """Creates the 3D U-Net model."""
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def train_one_epoch(model, loader, optimizer, scaler, device, loss_l1, loss_ssim, amp: bool):
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    n = 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            pred = model(src)
            l1 = loss_l1(pred, tgt)
            ssim = loss_ssim(pred, tgt)  # This is already 1-SSIM
            loss = l1 + 0.1 * ssim
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * src.size(0)
        n += src.size(0)
    return running_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, device):
    """Runs validation and returns metrics."""
    model.eval()
    mae_sum = 0.0
    ssim_sum = 0.0
    n = 0
    ssim_metric = SSIMLoss(spatial_dims=3)
    l1 = nn.L1Loss()
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        pred = model(src)
        mae = l1(pred, tgt).item()
        ssim = 1.0 - ssim_metric(pred, tgt).item()
        mae_sum += mae * src.size(0)
        ssim_sum += ssim * src.size(0)
        n += src.size(0)
    return mae_sum / max(1, n), ssim_sum / max(1, n)


def save_validation_preview(batch, output, save_dir, epoch):
    """Saves a 2D slice preview of the source, target, and prediction."""
    src = batch["src"][0, 0].cpu().numpy()
    tgt = batch["tgt"][0, 0].cpu().numpy()
    pred = output[0, 0].cpu().detach().numpy() # Use .detach() here

    z_slice = src.shape[2] // 2
    src_slice, tgt_slice, pred_slice = src[:, :, z_slice], tgt[:, :, z_slice], pred[:, :, z_slice]
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(axs, [src_slice, tgt_slice, pred_slice], ["Source", "Target", "Prediction"]):
        ax.imshow(data.T, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    
    fig.suptitle(f"Epoch {epoch} Validation Sample")
    plt.tight_layout()
    save_path = Path(save_dir) / f"epoch_{epoch:03d}_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train 3D UNet for MRI/CBCT → CT on SynthRad (128^3).")
    ap.add_argument("--splits", type=Path, required=True, help="dir with train.csv/val.csv")
    ap.add_argument("--exp-name", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", type=Path, default=Path("outputs/experiments"))
    ap.add_argument("--resume-from", type=Path, default=None, help="Path to a .pt checkpoint to continue training from.")
    args = ap.parse_args()

    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = (args.save_dir / args.exp_name).resolve()
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Note: cache_rate=1.0 is fastest but requires enough RAM to hold the whole dataset.
    # For standard Colab, a value of 0.5 might be safer.
    train_ds = CacheDataset(data=load_split(args.splits / "train.csv"), transform=make_transforms(train=True), cache_rate=1.0, num_workers=args.num_workers)
    val_ds = CacheDataset(data=load_split(args.splits / "val.csv"), transform=make_transforms(train=False), cache_rate=1.0, num_workers=args.num_workers)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device='cuda', enabled=args.amp)
    loss_l1 = nn.L1Loss()
    loss_ssim = SSIMLoss(spatial_dims=3)

    # --- UPGRADED: Logic for Resuming Full Training State ---
    best_mae = float("inf")
    start_epoch = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_mae = ckpt.get("best_mae", float("inf"))
        print(f"[INFO] Resumed training from checkpoint. Starting at epoch {start_epoch}.")
    
    # --- Training Loop Starts from the Correct Epoch ---
    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_l1, loss_ssim, amp=args.amp)
        val_mae, val_ssim = validate(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.max_epochs} | train_loss={train_loss:.4f} | val_MAE={val_mae:.4f} | val_SSIM={val_ssim:.4f} | {dt:.1f}s")

        # --- Save Best Checkpoint (with full state) ---
        if val_mae < best_mae:
            best_mae = val_mae
            best_ckpt_path = exp_dir / "checkpoints" / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mae": best_mae,
            }, best_ckpt_path)
            print(f"[INFO] Saved best checkpoint: {best_ckpt_path}")

            val_batch_for_preview = next(iter(val_loader))
            val_output = model(val_batch_for_preview["src"].to(device))
            save_validation_preview(val_batch_for_preview, val_output, exp_dir / "samples", epoch)
            print(f"[INFO] Saved validation preview for epoch {epoch}.")

        # --- Save Latest Checkpoint After EVERY Epoch ---
        last_ckpt_path = exp_dir / "checkpoints" / "last.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mae": best_mae,
        }, last_ckpt_path)

    print(f"[INFO] Training complete. Best val MAE: {best_mae:.4f} -> {exp_dir}")

if __name__ == "__main__":
    main()