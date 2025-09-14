#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
from typing import Tuple, Optional, List

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# -----------------------
# Model and helpers
# -----------------------
def make_model():
    """Creates the 3D U-Net model."""
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="INSTANCE",
    )

def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorients a NIfTI image to RAS+ orientation."""
    return nib.as_closest_canonical(img)

def spacing_from(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """Gets voxel spacing from a NIfTI image header."""
    z = img.header.get_zooms()[:3]
    return float(z[0]), float(z[1]), float(z[2])

def resample_iso1(vol: np.ndarray, spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    """Resamples a volume to 1mm isotropic spacing."""
    factors = np.array(spacing, dtype=np.float32)
    zoom_factors = 1.0 / np.clip(factors, 1e-6, 1e6)
    return zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1)).astype(np.float32)

def resample_from_iso1(vol: np.ndarray, original_spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    """Correctly resamples a 1mm isotropic volume back to its original spacing."""
    current_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    target_spacing = np.array(original_spacing, dtype=np.float32)
    # Corrected the clip upper bound to 1e6
    zoom_factors = current_spacing / np.clip(target_spacing, 1e-6, 1e6)
    return zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1)).astype(np.float32)

def normalize_mr_cbct(vol: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    """Performs robust percentile-based normalization for MR/CBCT."""
    finite = np.isfinite(vol)
    lo, hi = np.percentile(vol[finite], [p_low, p_high])
    if hi <= lo:
        m, s = float(vol.mean()), float(vol.std() + 1e-6)
        vol = (vol - m) / (3 * s) + 0.5
        return np.clip(vol, 0, 1).astype(np.float32)
    vol = np.clip(vol, lo, hi); vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)

def modality_from_filename(p: Path) -> str:
    """Infers modality from filename."""
    n = p.name.lower()
    if "cbct" in n: return "cbct"
    return "mr"

def save_preview_png(src: np.ndarray, pred: np.ndarray, out_png: Path, title: str = ""):
    """Saves a 3-panel preview of the inference result."""
    z = src.shape[2] // 2
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(np.rot90(src[:, :, z]), cmap="gray"); axs[0].set_title("Source"); axs[0].axis("off")
    axs[1].imshow(np.rot90(pred[:, :, z]), cmap="gray"); axs[1].set_title("Predicted sCT"); axs[1].axis("off")
    if title: fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

# -----------------------
# Core Inference Functions
# -----------------------

# --- CHANGE 1: Made `out_nii` an optional argument ---
def infer_single(model: torch.nn.Module, input_path: Path, out_nii: Optional[Path] = None, sw_size=(128, 128, 128),
                 overlap=0.25, amp=False, modality: Optional[str] = None):
    """
    Runs the full inference pipeline for a single case using a pre-loaded model.
    Saves NIfTI and PNG only if `out_nii` is provided.
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. Load and Preprocess
    img = as_ras(nib.load(str(input_path)))
    spacing = spacing_from(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    vol_r = resample_iso1(vol, spacing, order=1)
    src_mod = (modality or modality_from_filename(input_path)).lower()
    src_n = normalize_mr_cbct(vol_r)
    x = torch.from_numpy(src_n[None, None]).to(device)

    # 2. Run Inference
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        pred_iso = sliding_window_inference(x, roi_size=sw_size, sw_batch_size=1, predictor=model, overlap=overlap, mode="gaussian")
    
    pred_iso = torch.clamp(pred_iso, 0, 1)[0, 0].cpu().numpy().astype(np.float32)

    # 3. Post-process
    pred_resampled_back = resample_from_iso1(pred_iso, spacing)
    
    # --- CHANGE 2: Only save files if an output path is provided ---
    if out_nii:
        print(f"[INFO] Saving prediction to: {out_nii}")
        out_nii.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_resampled_back, img.affine), str(out_nii))
        
        # Also save the visual preview
        preview_path = out_nii.with_suffix("").with_suffix(".png")
        save_preview_png(vol, pred_resampled_back, preview_path,
                         title=f"{src_mod} → sCT | {input_path.name}")
        print(f"[INFO] Saved preview to: {preview_path}")

    return vol, pred_resampled_back, img


def main():
    ap = argparse.ArgumentParser(description="Inference CLI for SynthRad UNet-3D.")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt checkpoint.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single NIfTI source for inference.")
    group.add_argument("--csv", type=Path, help="CSV with 'src_nii' for batch inference.")
    ap.add_argument("--modality", type=str, default=None, help="mr or cbct (inferred if not set).")
    ap.add_argument("--out", type=Path, required=True, help="Output NIfTI file or directory.")
    ap.add_argument("--sw-size", type=int, nargs=3, default=(128, 128, 128))
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model once
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model = make_model().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("generator", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)
    
    if args.input:
        infer_single(
            model=model, input_path=args.input, out_nii=args.out,
            sw_size=tuple(args.sw_size), overlap=args.overlap, amp=args.amp,
            modality=args.modality
        )
    else:
        out_dir = args.out.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(args.csv)
        
        for _, row in df.iterrows():
            src = Path(row["src_nii"])
            task = row.get("task", "Task")
            pid = row.get("patient_id", Path(src).parent.name)
            mod = row.get("source_modality", modality_from_filename(src))
            out_nii = out_dir / task / f"{pid}_pred-{mod}.nii.gz"
            
            print(f"\n--- Processing: {pid} ---")
            infer_single(
                model=model, input_path=src, out_nii=out_nii,
                sw_size=tuple(args.sw_size), overlap=args.overlap, amp=args.amp,
                modality=mod
            )
        print(f"\n[INFO] Finished batch processing. Saved {len(df)} predictions to {out_dir}")

if __name__ == "__main__":
    main()

