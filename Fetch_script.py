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


# Attempts lazy imports of nibabel and pydicom, warning and returning early if they’re unavailable
# Searches for the first .nii.gz or .nii file; if found, loads it and prints its shape and data type
# If no NIfTI is found, looks for a .dcm file; reads and prints Modality, Rows, and Columns metadata
# If neither format is present or loading fails, logs an informational or warning message accordingly


def quick_peek(root: Path) -> None:
    try:
        _lazy_imports()
    except Exception:
        print("[WARN] nibabel/pydicom not available; skipping peek.")
        return
    nii = None
    for ext in ("*.nii.gz", "*.nii"):
        it = list(root.rglob(ext))
        if it:
            nii = it[0]
            break
    if nii:
        try:
            img = nib.load(str(nii))
            data = img.get_fdata(dtype="float32")
            print(f"\n[PEEK] NIfTI: {nii}")
            print(f"       shape={tuple(data.shape)}, dtype={data.dtype}")
            return
        except Exception as e:
            print(f"[WARN] Failed to read NIfTI {nii}: {e}")
    dcm = None
    it = list(root.rglob("*.dcm"))
    if it:
        dcm = it[0]
    if dcm:
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            print(f"\n[PEEK] DICOM: {dcm}")
            print(f"       Modality={getattr(ds, 'Modality', 'NA')}, Rows={getattr(ds, 'Rows', 'NA')}, Cols={getattr(ds, 'Columns', 'NA')}")
            return
        except Exception as e:
            print(f"[WARN] Failed to read DICOM {dcm}: {e}")
    print("[INFO] Could not peek any sample file (no NIfTI/DICOM found).")


# Lazy-imports gdown and logs the start of the download from the specified Drive folder
# Ensures the output directory and a temporary _gdrive_tmp subfolder exist
# Uses gdown.download_folder to fetch all files into the temp folder, then scans for archive files
# Moves each found archive from temp to the output directory (removing any existing ones), cleans up temp, and returns the moved archive list

def download_from_drive_folder(folder_id: str, dst: Path) -> List[Path]:
    _lazy_imports()
    print(f"[INFO] Downloading Google Drive folder {folder_id} into {dst}")
    ensure_dir(dst)
    tmp = dst / "_gdrive_tmp"
    ensure_dir(tmp)
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{folder_id}",
        output=str(tmp),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    archives: List[Path] = []
    for p, _, files in os.walk(tmp):
        for f in files:
            fp = Path(p) / f
            if f.lower().endswith(".zip") or f.lower().endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
                archives.append(fp)
    if not archives:
        print("[WARN] No archives found; folder may already be extracted.")
    moved: List[Path] = []
    for a in archives:
        target = dst / a.name
        if target.exists():
            target.unlink()
        shutil.move(str(a), str(target))
        moved.append(target)
    shutil.rmtree(tmp, ignore_errors=True)
    return moved



# • 	Performs an HTTP GET with stream=True,allow_redirects=True , and a timeout.
# • 	Calls r.raise_for_status()  to surface any HTTP errors immediately.
# • 	Reads the content-Length  header to determine total download size.
# • 	Opens the destination file in binary write mode.
# • 	Wraps the write loop in a tqdm progress bar (unit “B”, auto-scaled, labeled with the filename).
# • 	Iterates over r.iter_content(chunk_size=chunk) , writing each non-empty chunk to disk and updating the bar.
# • 	Leverages context managers to ensure the network connection and file handle close cleanly once done

def _download_file(url: str, out_path: Path, chunk: int = 1024 * 1024) -> None:
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as pbar:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)
                    pbar.update(len(b))



# - Prints an INFO line indicating which Zenodo record is being queried.
# - Ensures the output directory exists via ensure_dir.
# - Builds the Zenodo API URL (https://zenodo.org/api/records/{record_id}) and performs a GET with a timeout, then parses the JSON response to extract the list of available files.
# - If filename_contains filters are provided, applies them to the file list (case-insensitive substring match).
# - If no files remain after filtering, logs an error and returns an empty list.
# - Iterates over each file entry:
# - Determines the local target path as dst / file["key"].
# - If the file already exists on disk, skips downloading.
# - Otherwise, calls _download_file to stream the file with a tqdm progress bar.
# - If the filename’s extension matches common archive types (.zip, .tar.gz, .tgz, .tar.bz2, .tar.xz, .tar), appends its Path to the archives list.
# - Returns the list of downloaded archive Path objects.


def download_from_zenodo_record(record_id: str, dst: Path, filename_contains: Optional[List[str]] = None) -> List[Path]:
    print(f"[INFO] Querying Zenodo record {record_id}")
    ensure_dir(dst)
    api = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(api, timeout=60)
    r.raise_for_status()
    data = r.json()
    files = data.get("files", [])
    if filename_contains:
        files = [f for f in files if any(k.lower() in f["key"].lower() for k in filename_contains)]
    if not files:
        print("[ERROR] No files found to download from Zenodo.")
        return []
    archives: List[Path] = []
    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        out_path = dst / name
        if out_path.exists():
            print(f"[INFO] Skipping existing file: {name}")
        else:
            print(f"[INFO] Downloading {name} from Zenodo")
            _download_file(url, out_path)
        if name.lower().endswith(".zip") or name.lower().endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
            archives.append(out_path)
    return archives


# - Sets up an ArgumentParser with flags:
# - --source (drive or zenodo)
# - --drive-folder-id and --zenodo-record-id
# - --out for output directory
# - --extract toggle
# - --peek toggle
# - Resolves and ensures the output path exists.
# - Dispatches based on --source:
# - For drive, validates --drive-folder-id then calls download_from_drive_folder.
# - For zenodo, validates --zenodo-record-id then calls download_from_zenodo_record.
# - If --extract is set, passes the returned archives into extract_archives.
# - Calls summarize_tree on the output directory to list contents and counts.
# - If --peek is set, invokes quick_peek to inspect a sample file.
# - Prints a final “[INFO] Done.” message before exiting.

def main():
    ap = argparse.ArgumentParser(description="Fetch SynthRad data (Drive or Zenodo).")
    ap.add_argument("--source", choices=["drive", "zenodo"], required=True)
    ap.add_argument("--drive-folder-id", type=str, help="Google Drive folder ID")
    ap.add_argument("--zenodo-record-id", type=str, help="Zenodo record ID (e.g., 7260705)")
    ap.add_argument("--out", type=Path, default=Path("data/raw"), help="Output directory")
    ap.add_argument("--extract", action="store_true", help="Extract archives after download")
    ap.add_argument("--peek", action="store_true", help="Try reading one NIfTI or DICOM")
    args = ap.parse_args()

    out: Path = args.out.resolve()
    ensure_dir(out)

    archives: List[Path] = []
    if args.source == "drive":
        if not args.drive_folder_id:
            print("[ERROR] --drive-folder-id is required for source=drive")
            sys.exit(2)
        archives = download_from_drive_folder(args.drive_folder_id, out)
    elif args.source == "zenodo":
        if not args.zenodo_record_id:
            print("[ERROR] --zenodo-record-id is required for source=zenodo")
            sys.exit(2)
        archives = download_from_zenodo_record(args.zenodo_record_id, out)

    if args.extract:
        extract_archives(archives, out)

    summarize_tree(out)
    if args.peek:
        quick_peek(out)
    print("\n[INFO] Done.")


#Executes designated code exclusively when a module is launched as the main program, not when it’s imported.
if __name__ == "__main__":
    main()