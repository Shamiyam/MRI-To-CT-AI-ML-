#!/usr/bin/env python3
from multiprocessing import Pool
from functools import partial
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import zoom
from tqdm import tqdm


def as_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    # Reorient to RAS using nibabel
    return nib.as_closest_canonical(img)


def load_nii(p: Path) -> nib.Nifti1Image:
    return nib.load(str(p))


def spacing_from(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    z = img.header.get_zooms()
    return float(z[0]), float(z[1]), float(z[2])


def resample_iso1(data: np.ndarray, spacing: Tuple[float, float, float], order: int = 1) -> np.ndarray:
    # Compute zoom factors: old_spacing / new_spacing (new_spacing=1.0)
    factors = np.array(spacing, dtype=np.float32) / 1.0
    # Avoid degenerate values
    factors = np.clip(factors, 1e-6, 1e6)
    # scipy zoom expects zoom per axis as ratio new/old size, so we invert
    zoom_factors = 1.0 / factors
    out = zoom(data, zoom=zoom_factors, order=order, mode="constant", cval=0.0, prefilter=(order > 1))
    return out.astype(np.float32, copy=False)


def center_crop_or_pad(vol: np.ndarray, out_shape=(128, 128, 128)) -> np.ndarray:
    out_z, out_y, out_x = out_shape
    z, y, x = vol.shape
    result = np.zeros(out_shape, dtype=vol.dtype)

    # Compute start indices for crop in input space
    sz = max(0, (z - out_z) // 2)
    sy = max(0, (y - out_y) // 2)
    sx = max(0, (x - out_x) // 2)

    # Compute paste start indices in output space (for padding)
    dz = max(0, (out_z - z) // 2)
    dy = max(0, (out_y - y) // 2)
    dx = max(0, (out_x - x) // 2)

    # Compute extents to copy
    ez = min(z, out_z)
    ey = min(y, out_y)
    ex = min(x, out_x)

    result[dz:dz + ez, dy:dy + ey, dx:dx + ex] = vol[sz:sz + ez, sy:sy + ey, sx:sx + ex]
    return result


def normalize_ct(vol: np.ndarray, hu_min=-1000.0, hu_max=2000.0) -> np.ndarray:
    vol = np.clip(vol, hu_min, hu_max)
    return ((vol - hu_min) / (hu_max - hu_min)).astype(np.float32)


def normalize_mr_cbct(vol: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    # Robust min-max to [0,1]
    lo, hi = np.percentile(vol[np.isfinite(vol)], [p_low, p_high])
    if hi <= lo:
        m, s = float(vol.mean()), float(vol.std() + 1e-6)
        vol = (vol - m) / (3 * s) + 0.5
        return np.clip(vol, 0, 1).astype(np.float32)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)


def infer_src_mod(row: Dict) -> str:
    if "source_modality" in row and isinstance(row["source_modality"], str) and row["source_modality"]:
        return row["source_modality"].lower()
    # Fallback from filename
    n = str(row.get("source_path", "") or row.get("mr_path", "")).lower()
    if "cbct" in n:
        return "cbct"
    if any(k in n for k in ["mr", "mri", "t1", "t2", "flair"]):
        return "mr"
    return "unknown"


def get_src_path(row: Dict) -> str:
    for key in ["source_path", "src_path", "mr_path", "cbct_path"]:
        if key in row and isinstance(row[key], str) and row[key]:
            return row[key]
    raise KeyError("No source path column found (expected one of: source_path, src_path, mr_path, cbct_path).")

# The function below processes one patient at a time in a single loop. For massive datasets, we could speed this up significantly 
# by using Python's multiprocessing library to process multiple patients in parallel, 
# taking advantage of all the CPU cores available in one's Colab runtime.
"""def process_pair(row: Dict, out_dir: Path, out_shape=(128, 128, 128)) -> Dict:
    task = row.get("task", "")
    pid = row.get("patient_id", "")
    src_mod = infer_src_mod(row)
    src_path = Path(get_src_path(row))
    ct_path = Path(row["ct_path"])

    # Load
    src_img = as_ras(load_nii(src_path))
    ct_img = as_ras(load_nii(ct_path))

    # Extract data and spacing
    src = np.asarray(src_img.get_fdata(dtype=np.float32))
    ct = np.asarray(ct_img.get_fdata(dtype=np.float32))

    src_sp = spacing_from(src_img)
    ct_sp = spacing_from(ct_img)

    # Resample to 1mm iso
    src_r = resample_iso1(src, src_sp, order=1)
    ct_r = resample_iso1(ct, ct_sp, order=1)

    # Normalize
    if src_mod in ("mr", "cbct"):
        src_n = normalize_mr_cbct(src_r)
    else:
        # Rare case: if source is CT-like
        src_n = normalize_ct(src_r)
    ct_n = normalize_ct(ct_r)

    # Crop/pad to uniform shape
    src_f = center_crop_or_pad(src_n, out_shape)
    ct_f = center_crop_or_pad(ct_n, out_shape)

    # Save with standard naming
    base = f"{task}_{pid}".strip("_")
    src_name = f"{base}_src-{src_mod}.nii.gz"
    tgt_name = f"{base}_tgt-ct.nii.gz"

    pair_dir = out_dir / task / pid
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Use an identity RAS affine with 1mm spacing
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)

    nib.save(nib.Nifti1Image(src_f.astype(np.float32), affine), str(pair_dir / src_name))
    nib.save(nib.Nifti1Image(ct_f.astype(np.float32), affine), str(pair_dir / tgt_name))

    return dict(
        task=task,
        patient_id=pid,
        source_modality=src_mod,
        src_nii=str((pair_dir / src_name).resolve()),
        tgt_nii=str((pair_dir / tgt_name).resolve()),
        spacing_x=1.0,
        spacing_y=1.0,
        spacing_z=1.0,
        axes="RAS",
        shape=str(src_f.shape),
    ) """


#New function that support parallel processing-
# The only change is wrapping the existing code in a try/except block
def process_pair(row: Dict, out_dir: Path, out_shape=(128, 128, 128)) -> Dict:
    try:
        task = row.get("task", "")
        pid = row.get("patient_id", "")
        src_mod = infer_src_mod(row)
        src_path = Path(get_src_path(row))
        ct_path = Path(row["ct_path"])

        # ... (all your existing processing logic remains exactly the same) ...
        src_img = as_ras(load_nii(src_path))
        ct_img = as_ras(load_nii(ct_path))
        src = np.asarray(src_img.get_fdata(dtype=np.float32))
        ct = np.asarray(ct_img.get_fdata(dtype=np.float32))
        src_sp = spacing_from(src_img)
        ct_sp = spacing_from(ct_img)
        src_r = resample_iso1(src, src_sp, order=1)
        ct_r = resample_iso1(ct, ct_sp, order=1)
        if src_mod in ("mr", "cbct"):
            src_n = normalize_mr_cbct(src_r)
        else:
            src_n = normalize_ct(src_r)
        ct_n = normalize_ct(ct_r)
        src_f = center_crop_or_pad(src_n, out_shape)
        ct_f = center_crop_or_pad(ct_n, out_shape)
        base = f"{task}_{pid}".strip("_")
        src_name = f"{base}_src-{src_mod}.nii.gz"
        tgt_name = f"{base}_tgt-ct.nii.gz"
        pair_dir = out_dir / task / pid
        pair_dir.mkdir(parents=True, exist_ok=True)
        affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        nib.save(nib.Nifti1Image(src_f.astype(np.float32), affine), str(pair_dir / src_name))
        nib.save(nib.Nifti1Image(ct_f.astype(np.float32), affine), str(pair_dir / tgt_name))

        # On success, return the metadata dictionary
        return dict(
            task=task,
            patient_id=pid,
            source_modality=src_mod,
            src_nii=str((pair_dir / src_name).resolve()),
            tgt_nii=str((pair_dir / tgt_name).resolve()),
            spacing_x=1.0,
            spacing_y=1.0,
            spacing_z=1.0,
            axes="RAS",
            shape=str(src_f.shape),
        )
    except Exception as e:
        # On failure, return an error dictionary
        return dict(task=row.get("task", ""), patient_id=row.get("patient_id", ""), error=str(e))
    

# Old main fucntion for processing one patient at a time
"""def main():
    ap = argparse.ArgumentParser(description="Preprocess SynthRad pairs to RAS, 1mm iso, [0,1], 128^3.")
    ap.add_argument("--metadata", type=Path, required=True, help="CSV from EDA (must have ct_path and a source path).")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for processed NIfTI files.")
    ap.add_argument("--out-shape", type=int, nargs=3, default=(128, 128, 128), help="Output 3D shape.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of pairs for a dry-run (0 = all).")
    ap.add_argument("--save-zip", type=Path, default=None, help="If set, zip the out directory to this path.")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    rows = meta.to_dict(orient="records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = []
    errors = 0
    for row in tqdm(rows, desc="Preprocessing"):
        try:
            rec = process_pair(row, out_dir, out_shape=tuple(args.out_shape))
            processed.append(rec)
        except Exception as e:
            errors += 1
            rec = dict(task=row.get("task", ""), patient_id=row.get("patient_id", ""), error=str(e))
            processed.append(rec)

    proc_df = pd.DataFrame(processed)
    csv_path = out_dir / "processed_metadata.csv"
    proc_df.to_csv(csv_path, index=False)

    summary = dict(
        input_metadata=str(args.metadata.resolve()),
        out_dir=str(out_dir),
        pairs=len(rows),
        saved=int((proc_df["src_nii"].notna() & proc_df["tgt_nii"].notna()).sum()),
        errors=int(errors),
        out_shape=tuple(args.out_shape),
        spacing=(1.0, 1.0, 1.0),
        axes="RAS",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"- CSV: {csv_path}")

    if args.save_zip:
        zip_target = Path(args.save_zip)
        zip_target.parent.mkdir(parents=True, exist_ok=True)
        # Create a temp name for shutil.make_archive (it appends .zip)
        tmp_base = out_dir.parent / f"{out_dir.name}"
        # Ensure we zip only the processed folder contents
        archive = shutil.make_archive(str(tmp_base), "zip", root_dir=str(out_dir))
        shutil.move(archive, str(zip_target))
        print(f"[INFO] Wrote zip: {zip_target}")


if __name__ == "__main__":
    main()"""

#New main function that supports multiproccesing-
def main():
    ap = argparse.ArgumentParser(description="Preprocess SynthRad pairs to RAS, 1mm iso, [0,1], 128^3.")
    # ... (all your argparse lines remain exactly the same) ...
    ap.add_argument("--metadata", type=Path, required=True, help="CSV from EDA (must have ct_path and a source path).")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for processed NIfTI files.")
    ap.add_argument("--out-shape", type=int, nargs=3, default=(128, 128, 128), help="Output 3D shape.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of pairs for a dry-run (0 = all).")
    ap.add_argument("--save-zip", type=Path, default=None, help="If set, zip the out directory to this path.")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    rows = meta.to_dict(orient="records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- NEW MULTIPROCESSING LOGIC ---
    # Use all available CPU cores
    num_workers = os.cpu_count()
    print(f"[INFO] Using {num_workers} worker processes to preprocess {len(rows)} pairs.")

    # Create a "partial" function. This freezes the 'out_dir' and 'out_shape' arguments,
    # so the pool only needs to provide the 'row' for each patient.
    worker_func = partial(process_pair, out_dir=out_dir, out_shape=tuple(args.out_shape))

    processed = []
    # Create the pool of workers
    with Pool(processes=num_workers) as pool:
        # pool.imap_unordered processes the 'rows' iterable in parallel.
        # We wrap it with tqdm to create a live progress bar.
        for result in tqdm(pool.imap_unordered(worker_func, rows), total=len(rows), desc="Preprocessing"):
            processed.append(result)
    
    # --- END OF NEW LOGIC ---

    proc_df = pd.DataFrame(processed)
    csv_path = out_dir / "processed_metadata.csv"
    proc_df.to_csv(csv_path, index=False)

    # Robustly check for an 'error' column before accessing it
    if "error" in proc_df.columns:
        errors = proc_df["error"].notna().sum()
    else:
        errors = 0 # If the column doesn't exist, there were no errors.

    summary = dict(
        # ... (your summary dictionary logic remains the same) ...
        input_metadata=str(args.metadata.resolve()),
        out_dir=str(out_dir),
        pairs=len(rows),
        saved=int((proc_df["src_nii"].notna() & proc_df["tgt_nii"].notna()).sum()),
        errors=int(errors),
        out_shape=tuple(args.out_shape),
        spacing=(1.0, 1.0, 1.0),
        axes="RAS",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"- CSV: {csv_path}")

    if args.save_zip:
        # ... (your zipping logic remains exactly the same) ...
        zip_target = Path(args.save_zip)
        zip_target.parent.mkdir(parents=True, exist_ok=True)
        tmp_base = out_dir.parent / f"{out_dir.name}"
        archive = shutil.make_archive(str(tmp_base), "zip", root_dir=str(out_dir))
        shutil.move(archive, str(zip_target))
        print(f"[INFO] Wrote zip: {zip_target}")


if __name__ == "__main__":
    main()