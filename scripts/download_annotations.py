"""
Downloads pre-computed segment annotation files required by SAGE.

Segment annotations encode which frames belong to which sign segment.
Rather than re-running the sign segmentation model, we provide these
as ready-to-use pickle files.

After downloading, point sage.yaml at the files:
  data:
    segment_path: support_files/phoenix14T_segments.pkl   # or whichever dataset

Usage:
  python scripts/download_annotations.py --dataset phx14t --output_dir support_files/
  python scripts/download_annotations.py --dataset csldaily --output_dir support_files/
  python scripts/download_annotations.py --dataset all --output_dir support_files/
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry — update these URLs once files are hosted (HuggingFace Hub, etc.)
# ---------------------------------------------------------------------------
_ANNOTATIONS = {
    "phx14t": {
        "url":      "https://huggingface.co/datasets/SAGE-SLT/annotations/resolve/main/phoenix14T_segments.pkl",
        "filename": "phoenix14T_segments.pkl",
        "md5":      None,  # fill in after upload
    },
    "csldaily": {
        "url":      "https://huggingface.co/datasets/SAGE-SLT/annotations/resolve/main/csldaily_segments.pkl",
        "filename": "csldaily_segments.pkl",
        "md5":      None,
    },
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dataset: str, output_dir: Path):
    meta = _ANNOTATIONS[dataset]
    dest = output_dir / meta["filename"]

    if dest.exists():
        if meta["md5"] and _md5(dest) == meta["md5"]:
            print(f"[{dataset}] already downloaded and verified: {dest}")
            return
        print(f"[{dataset}] file exists but re-downloading (md5 mismatch or unchecked)")

    url = meta["url"]
    if "TODO" in url or "huggingface.co/datasets/SAGE-SLT" in url:
        print(
            f"[{dataset}] Download URL not yet set.\n"
            f"  Please obtain the file manually and place it at:\n"
            f"  {dest}\n"
            f"  (Expected URL placeholder: {url})"
        )
        return

    print(f"[{dataset}] Downloading {url} → {dest}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()

    if meta["md5"]:
        actual = _md5(dest)
        if actual != meta["md5"]:
            dest.unlink()
            raise RuntimeError(
                f"[{dataset}] MD5 mismatch: expected {meta['md5']}, got {actual}. "
                "File deleted — please try again."
            )
        print(f"[{dataset}] MD5 verified ✓")

    print(f"[{dataset}] Saved to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download SAGE segment annotations")
    parser.add_argument(
        "--dataset", required=True,
        choices=[*_ANNOTATIONS.keys(), "all"],
        help="Which dataset annotations to download (or 'all')",
    )
    parser.add_argument(
        "--output_dir", default="support_files",
        help="Directory to save annotation files (default: support_files/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    datasets = list(_ANNOTATIONS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        download(ds, output_dir)


if __name__ == "__main__":
    main()
