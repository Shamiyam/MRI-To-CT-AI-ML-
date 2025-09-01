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
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ToTensord
from monai.utils import set_determinism
import matplotlib.pyplot as plt


# -----------------------
# Model and helpers
# -----------------------
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


def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)


def spacing_from(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    z = img.header.get_zooms()[:3]
    return float(z[0]), float(z[1]), float(z[2])


def resample_iso1(vol: np.ndarray, spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    # scipy zoom factor = new_size / old_size; with target spacing=1mm => factor = 1/spacing
    from scipy.ndimage import zoom
    factors = np.array(spacing, dtype=np.float32)
    zoom_factors = 1.0 / np.clip(factors, 1e-6, 1e6)
    out = zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    return out.astype(np.float32, copy=False)


def normalize_ct(vol: np.ndarray, hu_min=-1000.0, hu_max=2000.0) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    return ((vol - hu_min) / (hu_max - hu_min)).astype(np.float32)


def normalize_mr_cbct(vol: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    finite = np.isfinite(vol)
    lo, hi = np.percentile(vol[finite], [p_low, p_high])
    if hi <= lo:
        m, s = float(vol.mean()), float(vol.std() + 1e-6)
        vol = (vol - m) / (3 * s) + 0.5
        return np.clip(vol, 0, 1).astype(np.float32)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def modality_from_filename(p: Path) -> str:
    n = p.name.lower()
    if "cbct" in n:
        return "cbct"
    if any(k in n for k in ["mr", "mri", "t1", "t2", "flair", "src-mr"]):
        return "mr"
    return "mr"  # default


def save_png_triplet(src: np.ndarray, pred: np.ndarray, out_png: Path, title: str = ""):
    z = src.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(src[:, :, z].T, cmap="gray", origin="lower"); axs[0].set_title("Source"); axs[0].axis("off")
    axs[1].imshow(pred[:, :, z].T, cmap="gray", origin="lower"); axs[1].set_title("Pred sCT"); axs[1].axis("off")
    axs[2].imshow(np.abs(pred[:, :, z] - src[:, :, z]).T, cmap="magma", origin="lower"); axs[2].set_title("|sCT-src|"); axs[2].axis("off")
    if title:
        fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def resample_from_iso1(vol: np.ndarray, original_spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    """Resamples a 1mm isotropic volume back to its original spacing."""
    from scipy.ndimage import zoom
    original_spacing = np.array(original_spacing, dtype=np.float32)
    zoom_factors = 1.0 / np.clip(original_spacing, 1e-6, 1e-6) # This is now inverted
    out = zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    return out.astype(np.float32, copy=False)

# -----------------------
# Single-case inference
# -----------------------
def infer_single(input_path: Path, checkpoint: Path, out_nii: Path, sw_size=(128, 128, 128),
                 overlap=0.25, amp=False, modality: Optional[str] = None, tta=False) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load + canonicalize
    img = as_ras(nib.load(str(input_path)))
    spacing = spacing_from(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    # Resample to 1mm iso
    vol_r = resample_iso1(vol, spacing, order=1)

    # Normalize
    src_mod = (modality or modality_from_filename(input_path)).lower()
    if src_mod in ("mr", "cbct"):
        src_n = normalize_mr_cbct(vol_r)
    else:
        src_n = normalize_ct(vol_r)

    # Prepare tensor
    x = torch.from_numpy(src_n[None, None]).to(device)

    # Model
    model = make_model().to(device)
    ckpt = torch.load(str(checkpoint), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Sliding-window inference
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        def fwd(inp):
            return torch.clamp(sliding_window_inference(inp, roi_size=sw_size, sw_batch_size=1, predictor=model,
                                                        overlap=overlap, mode="gaussian"), 0, 1)
        if not tta:
            y = fwd(x)
        else:
            # 3-axis flip TTA (8 variants)
            flips = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
            acc = torch.zeros_like(x)
            for fx, fy, fz in flips:
                xx = x
                if fx: xx = torch.flip(xx, dims=[2])
                if fy: xx = torch.flip(xx, dims=[3])
                if fz: xx = torch.flip(xx, dims=[4])
                yy = fwd(xx)
                if fx: yy = torch.flip(yy, dims=[2])
                if fy: yy = torch.flip(yy, dims=[3])
                if fz: yy = torch.flip(yy, dims=[4])
                acc += yy
            y = acc / len(flips)

    pred = y[0, 0].detach().cpu().numpy().astype(np.float32)

    print("[INFO] Resampling prediction back to original image space...")
    # Resample the 1mm prediction back to the original MR's voxel spacing
    pred_resampled_back = resample_from_iso1(pred, spacing)

    # Save in standardized RAS 1mm space (identity affine like processed set)
    affine_ras_1mm = np.eye(4, dtype=np.float32)
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred, affine_ras_1mm), str(out_nii))

    # Quick preview
    save_png_triplet(src_n, pred, out_nii.with_suffix("").with_suffix(".png"),
                     title=f"{src_mod} → sCT | {input_path.name}")
    return out_nii


# -----------------------
# Batch from CSV
# -----------------------
def infer_from_csv(csv_path: Path, checkpoint: Path, out_dir: Path, amp=False, tta=False,
                   sw_size=(128, 128, 128), overlap=0.25):
    df = pd.read_csv(csv_path)
    if "src_nii" not in df.columns:
        raise ValueError("CSV must contain src_nii column.")
    # Optional columns
    get_pid = lambda r: r.get("patient_id", Path(r["src_nii"]).parent.name) if hasattr(r, "get") else ""
    get_task = lambda r: r.get("task", "Task")
    get_mod = lambda r: r.get("source_modality", modality_from_filename(Path(r["src_nii"])))

    saved: List[Path] = []
    for _, row in df.iterrows():
        src = Path(row["src_nii"])
        task = row["task"] if "task" in row and isinstance(row["task"], str) else "Task"
        pid = row["patient_id"] if "patient_id" in row and isinstance(row["patient_id"], str) else Path(src).parent.name
        mod = row["source_modality"] if "source_modality" in row else modality_from_filename(src)
        out_nii = out_dir / task / f"{pid}_pred-{mod}.nii.gz"
        p = infer_single(src, checkpoint, out_nii, sw_size=sw_size, overlap=overlap, amp=amp, modality=mod, tta=tta)
        saved.append(p)
    return saved


def main():
    ap = argparse.ArgumentParser(description="Inference CLI for SynthRad UNet-3D (sliding-window + optional TTA).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single NIfTI source (MR/CBCT) for inference")
    group.add_argument("--csv", type=Path, help="CSV with 'src_nii' (e.g., test.csv or processed_metadata.csv)")
    ap.add_argument("--modality", type=str, default=None, help="mr or cbct (optional, inferred from filename if not set)")
    ap.add_argument("--out", type=Path, required=True, help="Output NIfTI (for --input) or output directory (for --csv)")
    ap.add_argument("--sw-size", type=int, nargs=3, default=(128, 128, 128), help="Sliding window ROI size")
    ap.add_argument("--overlap", type=float, default=0.25, help="Sliding window overlap [0,1)")
    ap.add_argument("--tta", action="store_true", help="Enable flip TTA (8-way). Slower, usually +0.5–1.0 PSNR.")
    ap.add_argument("--amp", action="store_true", help="Enable autocast")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_determinism(seed=args.seed)
    if args.input:
        infer_single(args.input, args.checkpoint, args.out, sw_size=tuple(args.sw_size),
                     overlap=args.overlap, amp=args.amp, modality=args.modality, tta=args.tta)
        print(f"[INFO] Saved: {args.out}")
    else:
        out_dir = args.out.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        files = infer_from_csv(args.csv, args.checkpoint, out_dir, amp=args.amp, tta=args.tta,
                               sw_size=tuple(args.sw_size), overlap=args.overlap)
        print(f"[INFO] Saved {len(files)} predictions under {out_dir}")


if __name__ == "__main__":
    main()