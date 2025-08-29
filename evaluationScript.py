#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from monai.networks.nets import UNet
from monai.losses import SSIMLoss
from monai.utils import set_determinism


# -----------------------
# Transforms & model
# -----------------------
def eval_transforms():
    return Compose([
        LoadImaged(keys=["src", "tgt"]),
        EnsureChannelFirstd(keys=["src", "tgt"]),
        ToTensord(keys=["src", "tgt"]),
    ])


def make_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def load_split_with_meta(csv_path: Path):
    df = pd.read_csv(csv_path)
    needed = {"src_nii", "tgt_nii"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Split CSV must contain columns: {needed}")
    # Keep extra metadata if present
    data = []
    for _, r in df.iterrows():
        data.append({
            "src": r["src_nii"],
            "tgt": r["tgt_nii"],
            "patient_id": r.get("patient_id", ""),
            "task": r.get("task", ""),
            "source_modality": r.get("source_modality", ""),
        })
    return data, df


# -----------------------
# Metrics
# -----------------------
def compute_metrics(pred: torch.Tensor, tgt: torch.Tensor):
    # pred, tgt: [B,1,D,H,W], values in [0,1]
    l1 = nn.L1Loss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")
    mae = l1(pred, tgt).item()
    msev = mse(pred, tgt).item()
    psnr = float("inf") if msev == 0 else 20.0 * math.log10(1.0) - 10.0 * math.log10(msev)
    ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)
    ssim = 1.0 - ssim_loss(pred, tgt).item()
    return mae, psnr, ssim


def clamp01(x: torch.Tensor):
    return torch.clamp(x, 0.0, 1.0)


# -----------------------
# Visualization & saving
# -----------------------
def save_prediction_nifti(pred_vol: np.ndarray, tgt_nii_path: str, out_path: Path):
    # Use the target affine to preserve spatial metadata (in processed set this is identity but consistent)
    tgt_img = nib.load(tgt_nii_path)
    pred_img = nib.Nifti1Image(pred_vol.astype(np.float32), tgt_img.affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pred_img, str(out_path))


def save_preview_png(src_vol: np.ndarray, pred_vol: np.ndarray, tgt_vol: np.ndarray, out_png: Path, title: str = ""):
    z = src_vol.shape[2] // 2
    s = src_vol[:, :, z]
    p = pred_vol[:, :, z]
    t = tgt_vol[:, :, z]
    diff = np.abs(t - p)

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(s.T, cmap="gray", origin="lower"); axs[0].set_title("Source"); axs[0].axis("off")
    axs[1].imshow(p.T, cmap="gray", origin="lower"); axs[1].set_title("Prediction"); axs[1].axis("off")
    axs[2].imshow(t.T, cmap="gray", origin="lower"); axs[2].set_title("Target"); axs[2].axis("off")
    im = axs[3].imshow(diff.T, cmap="magma", origin="lower"); axs[3].set_title("Abs Diff"); axs[3].axis("off")
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate 3D UNet on SynthRad test set.")
    ap.add_argument("--splits", type=Path, required=True, help="Directory with test.csv")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt")
    ap.add_argument("--save-dir", type=Path, default=Path("outputs/eval"))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save-nifti", action="store_true", help="Save NIfTI predictions")
    ap.add_argument("--save-previews", type=int, default=6, help="Number of PNG previews to save (top/worst/median).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_determinism(seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_root = args.save_dir.resolve()
    (save_root / "preds").mkdir(parents=True, exist_ok=True)
    (save_root / "previews").mkdir(parents=True, exist_ok=True)

    test_data, test_df = load_split_with_meta(args.splits / "test.csv")
    test_ds = Dataset(data=test_data, transform=eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model + checkpoint
    model = make_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Iterate
    rows = []
    ssim_metric = SSIMLoss(spatial_dims=3, data_range=1.0)
    l1 = nn.L1Loss(reduction="mean")
    mse = nn.MSELoss(reduction="mean")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=args.amp):
        idx = 0
        for batch in test_loader:
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)

            pred = clamp01(model(src))

            # per-sample metrics (batch-size may be 1; handle >1 safely)
            for b in range(src.size(0)):
                pid = batch.get("patient_id", [""])[b] if isinstance(batch.get("patient_id"), list) else batch.get("patient_id", [""])[b]
                task = batch.get("task", [""])[b] if isinstance(batch.get("task"), list) else batch.get("task", [""])[b]
                smod = batch.get("source_modality", [""])[b] if isinstance(batch.get("source_modality"), list) else batch.get("source_modality", [""])[b]

                pred_b = pred[b:b+1]
                tgt_b = tgt[b:b+1]

                mae = l1(pred_b, tgt_b).item()
                msev = mse(pred_b, tgt_b).item()
                psnr = float("inf") if msev == 0 else 20.0 * math.log10(1.0) - 10.0 * math.log10(msev)
                ssim = 1.0 - ssim_metric(pred_b, tgt_b).item()

                src_path = test_data[idx + b]["src"]
                tgt_path = test_data[idx + b]["tgt"]

                row = dict(
                    index=idx + b,
                    patient_id=pid,
                    task=task,
                    source_modality=smod,
                    src_path=src_path,
                    tgt_path=tgt_path,
                    mae=mae,
                    psnr=psnr,
                    ssim=ssim,
                )
                rows.append(row)

                # Save NIfTI if requested
                if args.save_nifti:
                    pred_np = pred_b[0, 0].detach().cpu().numpy()
                    out_nii = save_root / "preds" / task / f"{pid}_pred.nii.gz"
                    save_prediction_nifti(pred_np, tgt_path, out_nii)

            idx += src.size(0)

    metrics_df = pd.DataFrame(rows)
    csv_path = save_root / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    # Aggregates
    def agg(df):
        psnr_finite = df.loc[np.isfinite(df["psnr"]), "psnr"]
        
        return {
            "N": len(df),
            "MAE_mean": float(df["mae"].mean()),
            "MAE_std": float(df["mae"].std(ddof=0)),
            "PSNR_mean": float(psnr_finite.mean()),
            "SSIM_mean": float(df["ssim"].mean()),
        }

    overall = agg(metrics_df)
    by_task = metrics_df.groupby("task").apply(agg).to_dict()
    by_mod = metrics_df.groupby("source_modality").apply(agg).to_dict()

    summary = {
        "checkpoint": str(args.checkpoint),
        "overall": overall,
        "by_task": by_task,
        "by_source_modality": by_mod,
        "csv": str(csv_path),
    }
    with open(save_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    print(f"- overall: {overall}")
    for k, v in by_task.items():
        print(f"- task[{k}]: {v}")
    for k, v in by_mod.items():
        print(f"- modality[{k}]: {v}")
    print(f"- CSV: {csv_path}")
    print(f"- summary.json: {save_root/'summary.json'}")

    # Previews: best/worst/median by MAE
    k = min(args.save_previews, max(1, len(metrics_df)//3))
    sorted_df = metrics_df.sort_values("mae")
    selected = pd.concat([
        sorted_df.head(k),
        sorted_df.tail(k),
        sorted_df.iloc[[len(sorted_df)//2]]
    ]).drop_duplicates(subset=["patient_id"], keep="first")

    for _, r in selected.iterrows():
        # reload volumes for preview
        src_vol = nib.load(r["src_path"]).get_fdata().astype(np.float32)
        tgt_vol = nib.load(r["tgt_path"]).get_fdata().astype(np.float32)
        # if NIfTI prediction saved, load it; else re-run a quick forward for single case
        pred_path = save_root / "preds" / r["task"] / f"{r['patient_id']}_pred.nii.gz"
        if pred_path.exists():
            pred_vol = nib.load(str(pred_path)).get_fdata().astype(np.float32)
        else:
            x = torch.from_numpy(src_vol[None, None]).to(device)
            with torch.no_grad():
                y = torch.clamp(model(x), 0, 1)[0, 0].detach().cpu().numpy()
            pred_vol = y

        out_png = save_root / "previews" / f"{r['task']}_{r['patient_id']}_mae{r['mae']:.4f}.png"
        title = f"{r['task']} | {r['source_modality']} | MAE {r['mae']:.4f} | SSIM {r['ssim']:.3f}"
        save_preview_png(src_vol, pred_vol, tgt_vol, out_png, title)


if __name__ == "__main__":
    main()