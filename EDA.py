# - Shebang line (#!/usr/bin/env python3) makes the file directly executable as a Python script on UNIX systems.
# - Core utilities:
# - argparse for parsing command-line arguments
# - os and pathlib.Path for filesystem navigation
# - json and math for lightweight data serialization and math functions
# - Type hints (Dict, List, Optional, Tuple) improve readability and help IDEs/checkers catch errors.
# - Numerical/data libraries:
# - numpy and pandas for array operations and tabular dataframes
# - Medical-image I/O:
# - nibabel plus its aff2axcodes helper to load NIfTI files and extract spatial orientation codes
# - Visualization and feedback:
# - matplotlib.pyplot for generating preview figures
# - tqdm for a progress bar when scanning many patients


#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.orientations import aff2axcodes
import matplotlib.pyplot as plt
from tqdm import tqdm




# - is_nifti(p: Path) -> bool
# Checks if a filename ends with .nii or .nii.gz (case-insensitive), signalling a NIfTI image.

# - classify_modality(p: Path) -> str
# Inspects the lowercase filename for keywords:
# - Returns "ignore" if it looks like a segmentation or mask file (seg, mask, label).
# - Maps CBCT and CT tags to "ct".
# - Detects MRI-related tags (mri, mr, t1, t2, flair) as "mr".
# - Defaults to "unknown" otherwise.

# - find_patient_dirs(data_root: Path) -> List[Path]
# Recursively scans data_root for leaf directories containing at least one NIfTI file. Returns a sorted list of those patient folders.

# - pick_first(paths: List[Path], prefer: Optional[List[str]] = None) -> Optional[Path]
# From a list of file paths, returns the first match. If a prefer list is given, it prioritizes any path whose name contains one of the preferred substrings.


def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

#This included only MRI_CT pair so had to modify it.
"""def classify_modality(p: Path) -> str:
    n = p.name.lower()
    if any(k in n for k in ["seg", "mask", "label"]):
        return "ignore"
    if "cbct" in n:
        return "ct"  # treat CBCT as CT target (This is not what we wanted)
    if "ct" in n:
        return "ct"
    if any(k in n for k in ["mri", "mr", "t1", "t2", "flair"]):
        return "mr"
    return "unknown" """

#Modified version that would include CBCT as well-
def classify_modality(p: Path) -> str:
    """Classifies modality as mr, ct, cbct, or ignore."""
    n = p.name.lower()
    if any(k in n for k in ["seg", "mask", "label"]):
        return "ignore"
    if "cbct" in n:
        return "cbct"  # Now correctly identified as its own type
    if "ct" in n:
        return "ct"
    if any(k in n for k in ["mri", "mr", "t1", "t2", "flair"]):
        return "mr"
    return "unknown" 


def find_patient_dirs(data_root: Path) -> List[Path]:
    # Expect layout like: data_root/Task1/.../<patient_id>/files.nii.gz
    # We'll consider leaf dirs that contain at least one NIfTI
    candidates: List[Path] = []
    for p in data_root.rglob("*"):
        if p.is_dir():
            if any(is_nifti(f) for f in p.glob("*.nii*")):
                candidates.append(p)
    return sorted(candidates)


def pick_first(paths: List[Path], prefer: Optional[List[str]] = None) -> Optional[Path]:
    if not paths:
        return None
    if prefer:
        for key in prefer:
            for p in paths:
                if key in p.name.lower():
                    return p
    return paths[0]




# - load_header(path: Path)
#    • Loads the NIfTI via nibabel.
#    • Collapses 4D volumes by taking the first timepoint if needed.
#    • Extracts
#        - shape: the 3D array dimensions
#        - spacing: voxel sizes from the header zooms
#        - affine: the 4×4 spatial transform
#        - axcodes: human‐readable orientation codes (e.g., RAS)
#        - dtype: original data type
#            • Returns (img, shape, spacing, affine, axcodes, dtype) for downstream checks.
# - center_slice_2d(img: Nifti1Image) -> np.ndarray
# • Picks the middle axial slice (z = depth//2).
# • Converts to a NumPy array and zero‐fills any NaN or infinities.
# • Performs a robust intensity normalization:
# - Clips to the [1st, 99th] percentile range
# - Falls back to mean ± 3 std if percentiles collapse
# • Outputs a 2D float32 array in [0, 1] ready for display.

# - make_preview(mr_img, ct_img, out_png: Path)
# • Calls center_slice_2d on the MR and CT images.
# • Creates a side-by-side Matplotlib figure with grayscale MR/CT slices.
# • Removes axes, adds titles, and writes the PNG to out_png.

# - almost_equal(a: Tuple, b: Tuple, atol=1e-3) -> bool
# Compares two 3-element tuples (e.g., spacings) elementwise within an absolute tolerance, used to flag when MR and CT spacings or orientations diverge.


def load_header(path: Path):
    img = nib.load(str(path))
    shape = img.shape
    # Handle 4D: take first volume
    if len(shape) == 4 and shape[-1] > 1:
        shape = shape[:3]
    hdr = img.header
    spacing = tuple(float(z) for z in hdr.get_zooms()[:3])
    affine = img.affine
    axcodes = "".join(aff2axcodes(affine))
    dtype = str(img.get_data_dtype())
    return img, shape, spacing, affine, axcodes, dtype


def center_slice_2d(img: nib.Nifti1Image) -> np.ndarray:
    dataobj = img.dataobj  # lazy proxy
    shape = img.shape
    if len(shape) == 4:
        shape = shape[:3]
    z = shape[2] // 2
    sl = np.asarray(dataobj[:, :, z], dtype=np.float32)
    if np.isnan(sl).any() or np.isinf(sl).any():
        sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
    # Robust normalize to [0,1]
    p1, p99 = np.percentile(sl, [1, 99])
    if p99 > p1:
        sl = np.clip((sl - p1) / (p99 - p1), 0, 1)
    else:
        # fallback
        m, s = float(sl.mean()), float(sl.std() + 1e-6)
        sl = np.clip((sl - m) / (3 * s) + 0.5, 0, 1)
    return sl


def make_preview(mr_img: nib.Nifti1Image, ct_img: nib.Nifti1Image, out_png: Path) -> None:
    mr2d = center_slice_2d(mr_img)
    ct2d = center_slice_2d(ct_img)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(mr2d.T, cmap="gray", origin="lower")
    axs[0].set_title("MR")
    axs[0].axis("off")
    axs[1].imshow(ct2d.T, cmap="gray", origin="lower")
    axs[1].set_title("CT/CBCT")
    axs[1].axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def almost_equal(a: Tuple[float, float, float], b: Tuple[float, float, float], atol: float = 1e-3) -> bool:
    return all(abs(x - y) <= atol for x, y in zip(a, b))





# - It calls find_patient_dirs to get all leaf directories containing NIfTIs.
# - For each patient directory pdir:
# - Extracts the task name (e.g., Task1) and patient_id (the folder name).
# - Lists all NIfTI files (*.nii*).
# - Splits them into MR candidates and CT candidates via classify_modality.
# - Chooses one MR and one CT using pick_first (preferring “t1”, “mr” for MR and “ct”, “cbct” for CT).
# - Appends a dict with keys task, patient_id, mr_path, and ct_path (empty string if missing).
# Return value: a list of these dicts, one per patient directory.

# Old collect pairs did not consider CBCT so we edited and added new one that included it.
"""def collect_pairs(data_root: Path) -> List[Dict]:
    patient_dirs = find_patient_dirs(data_root)
    rows: List[Dict] = []
    for pdir in patient_dirs:
        task = pdir.parts[len(data_root.parts)] if len(pdir.parts) > len(data_root.parts) else ""
        patient_id = pdir.name
        niis = [f for f in pdir.glob("*.nii*") if is_nifti(f)]
        if not niis:
            continue
        mr_files = [f for f in niis if classify_modality(f) == "mr"]
        ct_files = [f for f in niis if classify_modality(f) == "ct"]

        mr = pick_first(mr_files, prefer=["t1", "mr", "mri"])
        ct = pick_first(ct_files, prefer=["ct", "cbct"])

        rows.append(
            dict(
                task=task,
                patient_id=patient_id,
                mr_path=str(mr) if mr else "",
                ct_path=str(ct) if ct else "",
            )
        )
    return rows """

def collect_pairs(data_root: Path) -> List[Dict]:
    """
    Finds MR-CT or CBCT-CT pairs.
    """
    patient_dirs = find_patient_dirs(data_root)
    rows: List[Dict] = []
    for pdir in patient_dirs:
        task = pdir.parts[len(data_root.parts)] if len(pdir.parts) > len(data_root.parts) else ""
        patient_id = pdir.name
        niis = [f for f in pdir.glob("*.nii*") if is_nifti(f)]
        if not niis:
            continue

        # Classify all available files
        mr_files = [f for f in niis if classify_modality(f) == "mr"]
        ct_files = [f for f in niis if classify_modality(f) == "ct"]
        cbct_files = [f for f in niis if classify_modality(f) == "cbct"]

        ct = pick_first(ct_files)
        source, source_modality = (None, None)

        # Prioritize MR as the source, but fall back to CBCT
        if mr_files:
            source = pick_first(mr_files, prefer=["t1", "mr", "mri"])
            source_modality = "MR"
        elif cbct_files:
            source = pick_first(cbct_files)
            source_modality = "CBCT"

        rows.append(
            dict(
                task=task,
                patient_id=patient_id,
                source_path=str(source) if source else "",
                ct_path=str(ct) if ct else "",
                source_modality=source_modality if source_modality else "",
            )
        )
    return rows





"""def eda(data_root: Path, out_dir: Path, max_previews: int = 16) -> Path:
    # The eda function takes:
    # - data_root: folder containing MR/CT files
    # - out_dir: where we’ll save CSV, JSON, and previews
    # - max_previews: cap on preview images (default 16)

    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir / "previews"
    # Prepare to collect per-patient metadata and count outcomes

    meta_rows: List[Dict] = []
    # Build a list of MR/CT candidate pairs, then loop with a progress bar

    pair_rows = collect_pairs(data_root)
    ok_pairs = 0
    missing_mr = 0
    missing_ct = 0
    affine_mismatch = 0

    for i, row in enumerate(tqdm(pair_rows, desc="Scanning patients")):
        mr_path = row["mr_path"]
        ct_path = row["ct_path"]
        if not mr_path:
            missing_mr += 1
            continue

        # If MR or CT is missing, update counters and skip this patient

        if not ct_path:
            missing_ct += 1
            continue

        try:
            # Load image headers and data for MR & CT
            # Compare shape, voxel spacing (tolerance 1e-3), and affine transforms


            mr_img, mr_shape, mr_spacing, mr_affine, mr_axes, mr_dtype = load_header(Path(mr_path))
            ct_img, ct_shape, ct_spacing, ct_affine, ct_axes, ct_dtype = load_header(Path(ct_path))

            aligned_shape = mr_shape == ct_shape
            aligned_spacing = almost_equal(mr_spacing, ct_spacing, atol=1e-3)
            aligned_affine = np.allclose(mr_affine, ct_affine, atol=1e-3)
            if not (aligned_shape and aligned_spacing and aligned_affine):
                affine_mismatch += 1

            # Extract center 2D slices and compute 1st & 99th intensity percentiles
            mr_cs = center_slice_2d(mr_img)
            ct_cs = center_slice_2d(ct_img)
            mr_p1, mr_p99 = float(np.percentile(mr_cs, 1)), float(np.percentile(mr_cs, 99))
            ct_p1, ct_p99 = float(np.percentile(ct_cs, 1)), float(np.percentile(ct_cs, 99))
            
            # Store all metadata into meta_rows
            meta_rows.append(
                dict(
                    task=row["task"],
                    patient_id=row["patient_id"],
                    mr_path=mr_path,
                    ct_path=ct_path,
                    mr_shape=str(mr_shape),
                    ct_shape=str(ct_shape),
                    mr_spacing_x=mr_spacing[0],
                    mr_spacing_y=mr_spacing[1],
                    mr_spacing_z=mr_spacing[2],
                    ct_spacing_x=ct_spacing[0],
                    ct_spacing_y=ct_spacing[1],
                    ct_spacing_z=ct_spacing[2],
                    mr_axes=mr_axes,
                    ct_axes=ct_axes,
                    mr_dtype=mr_dtype,
                    ct_dtype=ct_dtype,
                    aligned_shape=bool(aligned_shape),
                    aligned_spacing=bool(aligned_spacing),
                    aligned_affine=bool(aligned_affine),
                    mr_p1=mr_p1,
                    mr_p99=mr_p99,
                    ct_p1=ct_p1,
                    ct_p99=ct_p99,
                )
            )
            

            # If under the preview limit, create side-by-side MR/CT PNG

            if i < max_previews:
                out_png = previews_dir / f"{row['task']}_{row['patient_id']}.png"
                make_preview(mr_img, ct_img, out_png)

            ok_pairs += 1

        except Exception as e:
            # If anything goes wrong during loading or stats, record the error message

            meta_rows.append(
                dict(
                    task=row["task"],
                    patient_id=row["patient_id"],
                    mr_path=mr_path,
                    ct_path=ct_path,
                    error=str(e),
                )
            )

    #Convert metadata list to DataFrame and write CSV

    meta_df = pd.DataFrame(meta_rows)
    csv_path = out_dir / "synthrad_metadata.csv"
    meta_df.to_csv(csv_path, index=False)
    
    # Build and dump a lightweight summary JSON for quick checks

    summary = dict(
        data_root=str(data_root),
        patients_scanned=len(pair_rows),
        ok_pairs=ok_pairs,
        missing_mr=missing_mr,
        missing_ct=missing_ct,
        affine_or_shape_mismatch=affine_mismatch,
        previews_written=min(max_previews, ok_pairs),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


    # Print a neat console summary before returning the CSV path
    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"- CSV: {csv_path}")
    print(f"- Previews: {previews_dir} (up to {max_previews})")
    return csv_path """


#New EDA with minimum changes so that CBCT_pair is included as well

def eda(data_root: Path, out_dir: Path, max_previews: int = 16) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = out_dir / "previews"
    meta_rows: List[Dict] = []
    
    pair_rows = collect_pairs(data_root)
    
    # UPDATED: Counters for the new logic
    ok_pairs = 0
    missing_source = 0
    missing_ct = 0
    affine_mismatch = 0

    for i, row in enumerate(tqdm(pair_rows, desc="Scanning patients")):
        # CHANGED: from mr_path to source_path
        source_path = row["source_path"]
        ct_path = row["ct_path"]
        
        if not source_path:
            missing_source += 1 # CHANGED: counter name
            continue
        if not ct_path:
            missing_ct += 1
            continue

        try:
            # CHANGED: All 'mr_' variables renamed to 'source_'
            source_img, source_shape, source_spacing, source_affine, source_axes, source_dtype = load_header(Path(source_path))
            ct_img, ct_shape, ct_spacing, ct_affine, ct_axes, ct_dtype = load_header(Path(ct_path))

            aligned_shape = source_shape == ct_shape
            aligned_spacing = almost_equal(source_spacing, ct_spacing, atol=1e-3)
            aligned_affine = np.allclose(source_affine, ct_affine, atol=1e-3)
            if not (aligned_shape and aligned_spacing and aligned_affine):
                affine_mismatch += 1

            source_cs = center_slice_2d(source_img)
            ct_cs = center_slice_2d(ct_img)
            source_p1, source_p99 = float(np.percentile(source_cs, 1)), float(np.percentile(source_cs, 99))
            ct_p1, ct_p99 = float(np.percentile(ct_cs, 1)), float(np.percentile(ct_cs, 99))
            
            # UPDATED: Storing the new columns in the final CSV
            meta_rows.append(
                dict(
                    task=row["task"],
                    patient_id=row["patient_id"],
                    source_modality=row["source_modality"], # ADDED: new column
                    source_path=source_path,               # CHANGED: variable name
                    ct_path=ct_path,
                    source_shape=str(source_shape),        # CHANGED: variable name
                    ct_shape=str(ct_shape),
                    # ... all other columns remain, just with 'source_' instead of 'mr_'
                    source_spacing_x=source_spacing[0],
                    source_spacing_y=source_spacing[1],
                    source_spacing_z=source_spacing[2],
                    ct_spacing_x=ct_spacing[0],
                    ct_spacing_y=ct_spacing[1],
                    ct_spacing_z=ct_spacing[2],
                    source_axes=source_axes,
                    ct_axes=ct_axes,
                    source_dtype=source_dtype,
                    ct_dtype=ct_dtype,
                    aligned_shape=bool(aligned_shape),
                    aligned_spacing=bool(aligned_spacing),
                    aligned_affine=bool(aligned_affine),
                    source_p1=source_p1,
                    source_p99=source_p99,
                    ct_p1=ct_p1,
                    ct_p99=ct_p99,
                )
            )
            
            if i < max_previews:
                out_png = previews_dir / f"{row['task']}_{row['patient_id']}_{row['source_modality']}.png"
                make_preview(source_img, ct_img, out_png) # CHANGED: pass source_img

            ok_pairs += 1

        except Exception as e:
            meta_rows.append(dict(task=row["task"], patient_id=row["patient_id"], error=str(e)))

    meta_df = pd.DataFrame(meta_rows)
    csv_path = out_dir / "synthrad_metadata_all.csv" # New filename
    meta_df.to_csv(csv_path, index=False)
    
    # UPDATED: The summary dictionary
    summary = dict(
        data_root=str(data_root),
        patients_scanned=len(pair_rows),
        ok_pairs=ok_pairs,
        missing_source=missing_source,
        missing_ct=missing_ct,
        affine_or_shape_mismatch=affine_mismatch,
        previews_written=min(max_previews, ok_pairs),
    )
    with open(out_dir / "summary_all.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")
    print(f"- CSV: {csv_path}")
    print(f"- Previews: {previews_dir} (up to {max_previews})")
    return csv_path



# - Argument Parser Setup
# - Creates an ArgumentParser with a concise description of the EDA task.
# - Adds three flags:
# - --data-root (required): the top‐level folder containing your extracted Task1/Task2 patient data.
# - --out (optional, default outputs/eda): where to write the CSV, JSON, and preview images.
# - --max-previews (optional, default 16): how many MR/CT preview PNGs to generate.

def main():
    ap = argparse.ArgumentParser(description="EDA for SynthRad dataset: pair MR↔CT, check spacing/orientation, write CSV + previews.")
    ap.add_argument("--data-root", type=Path, required=True, help="Root folder that contains Task1/Task2/... (extracted).")
    ap.add_argument("--out", type=Path, default=Path("outputs/eda"), help="Output directory for CSV and previews.")
    ap.add_argument("--max-previews", type=int, default=16, help="Number of PNG previews to save.")
    args = ap.parse_args()
    #- Calls ap.parse_args() to read the user’s inputs.


    # - Resolves both data_root and out_dir to absolute paths.
    # - Checks that data_root actually exists; if not, immediately raises a FileNotFoundError with a helpful message.
    data_root = args.data_root.resolve()
    out_dir = args.out.resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    

    #Invokes , kicking off all the work you’ve already defined: pairing, header checks, CSV/JSON output, and previews.
    eda(data_root, out_dir, max_previews=args.max_previews)


if __name__ == "__main__":
    main()
    