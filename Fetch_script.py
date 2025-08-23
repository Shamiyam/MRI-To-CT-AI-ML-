#This part contains standard libraries for file handling, CLI parsing and HTTP requests
#tqdm adds a progress bar for donwloads
#Path is preferred over raw strings for file paths which is cleaner and safer
import argparse,os,shutil,sys,tarfile,zipfile
from pathlib import Path
from typing import List, Optional
import requests
from tqdm import tqdm


#We are using lazy imports because these imports are only needed if certain features
#are used.(eg peeking into medical files). Keeps startup fast and avoids crashing if option packages aren't installed.

def _lazy_imports():
    global gdown,nib,pydicom
    import importlib
    gdown=importlib.import_module("gdown")
    nib=importlib.import_module("nibabel")
    pydicom=importlib.import_module("pydicom")

def ensure_dir(P:Path) -> None:
    if not P.exists():
        P.mkdir(parents=True, exist_ok=True) #Creates a directory if it doesn't exist


# This function securely extracts tar archives by validating each file’s path.
# It prevents path traversal attacks by ensuring no file escapes the target directory.
# If a file’s resolved path isn’t within the intended extraction path, it raises an exception.
# This adds a critical layer of safety when handling untrusted tar files.

def safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory
    for member in tar.getmembers():
        member_path = path / member.name
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path)


# This function loops through a list of archive files and extracts them into a target directory.
# It handles , , , and other common formats, using  for security.
# Unknown formats are skipped with a warning, and errors during extraction are caught and logged.
# It prints progress and completion messages to keep the user informed.

def extract_archives(archives: List[Path], out_dir: Path) -> None:
    if not archives:
        print("[INFO] Nothing to extract.")
        return
    print(f"[INFO] Extracting {len(archives)} archive(s) into {out_dir}")
    for arc in archives:
        try:
            if arc.suffix.lower() == ".zip":
                with zipfile.ZipFile(arc, "r") as zf:
                    zf.extractall(out_dir)
            elif arc.suffixes[-2:] == [".tar", ".gz"] or arc.suffix.lower() in [".tgz", ".tar", ".tbz2", ".txz"]:
                with tarfile.open(arc, "r:*") as tf:
                    safe_extract_tar(tf, out_dir)
            else:
                print(f"[WARN] Skipping unknown archive type: {arc.name}")
                continue
            print(f"[INFO] Extracted: {arc.name}")
        except Exception as e:
            print(f"[ERROR] Failed to extract {arc.name}: {e}")
    print("[INFO] Extraction complete.")



# Prints a header showing top-level contents of root.
# Iterates over root, prefixing each entry with [D] for directories or [F] for files.
# Uses root.rglob()  to count all .nii /nii.gz and.dcm  files in the subtree.
# Prints a summary line with total NIfTI and DICOM file counts.

def summarize_tree(root: Path) -> None:
    print(f"\n[INFO] Top-level contents of {root}:")
    for entry in sorted(root.iterdir()):
        print(f"  [{'D' if entry.is_dir() else 'F'}] {entry.name}")
    nii = sum(1 for _ in root.rglob("*.nii")) + sum(1 for _ in root.rglob("*.nii.gz"))
    dcm = sum(1 for _ in root.rglob("*.dcm"))
    print(f"\n[INFO] Counts -> NIfTI: {nii}, DICOM: {dcm}")


