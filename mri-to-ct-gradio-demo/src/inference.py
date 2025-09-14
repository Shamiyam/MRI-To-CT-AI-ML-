#!/usr/bin/env python3
import argparse
from pathlib import Path
import math
from typing import Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from scipy.ndimage import zoom

def make_model():
    """Creates the 3D U-Net model."""
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm="INSTANCE",
    )

def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.as_closest_canonical(img)

def spacing_from(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    z = img.header.get_zooms()[:3]
    return float(z[0]), float(z[1]), float(z[2])

def resample_iso1(vol: np.ndarray, spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    factors = np.array(spacing, dtype=np.float32)
    zoom_factors = 1.0 / np.clip(factors, 1e-6, 1e6)
    return zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1)).astype(np.float32)

def normalize_mr_cbct(vol: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    finite = np.isfinite(vol)
    lo, hi = np.percentile(vol[finite], [p_low, p_high])
    if hi <= lo:
        m, s = float(vol.mean()), float(vol.std() + 1e-6)
        vol = (vol - m) / (3 * s) + 0.5
        return np.clip(vol, 0, 1).astype(np.float32)
    vol = np.clip(vol, lo, hi); vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)

def modality_from_filename(p: Path) -> str:
    n = p.name.lower()
    if "cbct" in n: return "cbct"
    return "mr"

def resample_from_iso1(vol: np.ndarray, original_spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    """Correctly resamples a 1mm isotropic volume back to its original spacing."""
    current_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    target_spacing = np.array(original_spacing, dtype=np.float32)
    zoom_factors = current_spacing / np.clip(target_spacing, 1e-6, 1e6)
    return zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1)).astype(np.float32)

def infer_single(model: torch.nn.Module, input_path: Path, sw_size=(128, 128, 128), overlap=0.25, amp=False, modality: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Runs inference using a pre-loaded model."""
    device = next(model.parameters()).device
    model.eval()

    img = as_ras(nib.load(str(input_path)))
    spacing = spacing_from(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    vol_r = resample_iso1(vol, spacing, order=1)

    src_mod = (modality or modality_from_filename(input_path)).lower()
    src_n = normalize_mr_cbct(vol_r)
    
    x = torch.from_numpy(src_n[None, None]).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        y = sliding_window_inference(x, roi_size=sw_size, sw_batch_size=1, predictor=model, overlap=overlap, mode="gaussian")
    
    pred_iso = torch.clamp(y, 0, 1)[0, 0].cpu().numpy().astype(np.float32)
    pred_resampled_back = resample_from_iso1(pred_iso, spacing)

    return vol, pred_resampled_back, img
