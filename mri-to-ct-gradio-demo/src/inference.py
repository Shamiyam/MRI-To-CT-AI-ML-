# filepath: /mri-to-ct-gradio-demo/src/inference.py
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from typing import Tuple, Optional


def make_model():
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
    from scipy.ndimage import zoom
    factors = np.array(spacing, dtype=np.float32)
    zoom_factors = 1.0 / np.clip(factors, 1e-6, 1e6)
    out = zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    return out.astype(np.float32, copy=False)

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
    if "cbct" in n: return "cbct"
    return "mr"

# --- THIS FUNCTION IS NOW CORRECTED ---
def resample_from_iso1(vol: np.ndarray, original_spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    """Resamples a 1mm isotropic volume back to its original spacing."""
    from scipy.ndimage import zoom
    current_spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    target_spacing = np.array(original_spacing, dtype=np.float32)
    zoom_factors = current_spacing / np.clip(target_spacing, 1e-6, 1e6) # Corrected upper bound
    out = zoom(vol, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    return out.astype(np.float32, copy=False)


# --- THIS FUNCTION IS NOW UPDATED TO TAKE THE MODEL AS AN ARGUMENT ---
def infer_single(model: torch.nn.Module, input_path: Path, sw_size=(128, 128, 128), overlap=0.25, amp=False, modality: Optional[str] = None, tta=False) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    device = next(model.parameters()).device

    img = as_ras(nib.load(str(input_path)))
    spacing = spacing_from(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    vol_r = resample_iso1(vol, spacing, order=1)

    src_mod = (modality or modality_from_filename(input_path)).lower()
    src_n = normalize_mr_cbct(vol_r) if src_mod in ("mr", "cbct") else normalize_ct(vol_r)
    
    x = torch.from_numpy(src_n[None, None]).to(device)

    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
        def fwd(inp):
            return torch.clamp(sliding_window_inference(inp, roi_size=sw_size, sw_batch_size=1, predictor=model, overlap=overlap, mode="gaussian"), 0, 1)
        
        y = fwd(x) # Simplified for demo, TTA can be added back if needed
    
    pred = y[0, 0].detach().cpu().numpy().astype(np.float32)
    pred_resampled_back = resample_from_iso1(pred, spacing)

    return vol, pred_resampled_back, img