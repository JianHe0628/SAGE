"""
Generates SAGE segment annotation files from raw segmentor output (.pt files).

The raw segmentor output labels each frame with:
  0 = transition / silence
  1 = inside a sign segment
  2 = start of a new sign segment (boundary marker)

Aggressive segmentation applied:
  - Splits at every frame labelled 2
  - Drops segments shorter than --min_len frames (default 4)
  - Recursively halves segments longer than --max_len frames (default 10)
  - Drops leading/trailing all-zero segments (pure transitions)

Output structure (same for both datasets):
  {
    "train": { video_id -> {
        "Frame_Range": [[frame_ref, ...], ...],
        "Translation": str,
    }},
    "dev":  { ... },
    "test": { ... },
  }

  PHX14T   Frame_Range: image path strings  e.g. ['train/VID/images0001.png', ...]
  CSLDaily Frame_Range: integer frame indices e.g. [0, 1, 2, ...]

The raw .pt files are shipped in support_files/segment_annotations/:
  phoenix14T_segmentor_outs.pt  — structure: {phase: {video_id: tensor}}
  csldaily_segmentor_outs.pt    — structure: {phase: {video_id: tensor}}

The --metadata file is the output of prepare_metadata.py, which embeds
video-to-translation mappings alongside the pseudo-gloss content.

Usage:
  # Phoenix-2014T
  python scripts/prepare_segment_annotations.py phx14t \
    --segs support_files/segment_annotations/SAGE/phoenix14T_segmentor_outs.pt \
    --metadata support_files/metadata/phoenix14T_metadata.pkl \
    --output support_files/phoenix14T_segments.pkl

  # CSL-Daily
  python scripts/prepare_segment_annotations.py csldaily \
    --segs support_files/segment_annotations/SAGE/csldaily_segmentor_outs.pt \
    --metadata support_files/metadata/csldaily_metadata.pkl \
    --output support_files/csldaily_segments.pkl
"""

import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm


def _split_half(indices, max_len):
    """Recursively halve a list of frame indices until every piece is <= max_len."""
    if len(indices) <= max_len:
        return [indices]
    mid = len(indices) // 2
    return _split_half(indices[:mid], max_len) + _split_half(indices[mid:], max_len)


def break_segments(label_tensor, min_len, max_len):
    """
    Convert a per-frame label tensor into a list of frame-index lists.

    Returns one list of integer indices per sign segment.
    """
    labels = label_tensor.tolist() if hasattr(label_tensor, 'tolist') else list(label_tensor)
    n = len(labels)

    if sum(labels) == 0:
        return _split_half(list(range(n)), max_len)

    breakpoints = sorted(set([0] + [i for i, l in enumerate(labels) if l == 2] + [n]))
    raw_segs = [(labels[s:e], list(range(s, e)))
                for s, e in zip(breakpoints, breakpoints[1:])]

    final = []
    total = len(raw_segs)
    for idx, (seg_labels, frames) in enumerate(raw_segs):
        if len(frames) < min_len:
            continue
        if (idx == 0 or idx == total - 1) and all(l == 0 for l in seg_labels):
            continue
        if len(frames) > max_len:
            final.extend(_split_half(frames, max_len))
        else:
            final.append(frames)
    return final


def _load_video_meta(metadata_path):
    """Load video_meta from the metadata pkl (output of prepare_metadata.py)."""
    with open(metadata_path, "rb") as f:
        meta = pickle.load(f)
    if "video_meta" not in meta:
        raise KeyError(
            f"'video_meta' not found in {metadata_path}. "
            "Re-run prepare_metadata.py to regenerate the metadata file."
        )
    return meta["video_meta"]  # {phase: {vid: {"translation": str}}}


def _process(segs_path, metadata_path, frame_range_fn, min_len, max_len, desc):
    print(f"Loading {desc} segmentor output...")
    segs = torch.load(segs_path, map_location='cpu', weights_only=False)

    print("Loading video metadata...")
    video_meta = _load_video_meta(metadata_path)

    result = {}
    for phase in ('train', 'dev', 'test'):
        phase_segs = segs.get(phase, {})
        phase_meta = video_meta.get(phase, {})
        phase_result = {}
        for vid, labels in tqdm(phase_segs.items(), desc=f"{desc} [{phase}]"):
            vid_info = phase_meta.get(vid)
            if vid_info is None:
                continue
            idx_lists = break_segments(labels, min_len, max_len)
            if not idx_lists:
                continue
            phase_result[vid] = {
                'Frame_Range': frame_range_fn(phase, vid, idx_lists),
                'Translation': vid_info["translation"],
            }
        result[phase] = phase_result
        n_segs = sum(len(v['Frame_Range']) for v in phase_result.values())
        print(f"  {phase}: {len(phase_result)} videos, {n_segs} segments")
    return result


def process_phx14t(args):
    def frame_range_fn(phase, vid, idx_lists):
        # PHX14T LMDB expects image path strings (1-indexed filenames)
        return [
            [f"{phase}/{vid}/images{i + 1:04d}.png" for i in seg]
            for seg in idx_lists
        ]
    return _process(args.segs, args.metadata, frame_range_fn,
                    args.min_len, args.max_len, "PHX14T")


def process_csldaily(args):
    def frame_range_fn(phase, vid, idx_lists):
        # CSLDaily LMDB expects integer frame indices
        return idx_lists
    return _process(args.segs, args.metadata, frame_range_fn,
                    args.min_len, args.max_len, "CSLDaily")


_RUNNERS = {"phx14t": process_phx14t, "csldaily": process_csldaily}


def main():
    parser = argparse.ArgumentParser(description="Generate SAGE segment annotation files")
    sub = parser.add_subparsers(dest="dataset", required=True)

    for name, help_str in (("phx14t", "Phoenix-2014T"), ("csldaily", "CSL-Daily")):
        p = sub.add_parser(name, help=help_str)
        p.add_argument("--segs", required=True,
                        help="Raw segmentor output .pt (from support_files/segment_annotations/)")
        p.add_argument("--metadata", required=True,
                        help="Metadata .pkl from prepare_metadata.py")
        p.add_argument("--output", required=True, help="Output .pkl path")
        p.add_argument("--min_len", type=int, default=4,
                        help="Drop segments shorter than this many frames (default: 4)")
        p.add_argument("--max_len", type=int, default=10,
                        help="Recursively halve segments longer than this (default: 10)")

    args = parser.parse_args()
    result = _RUNNERS[args.dataset](args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    total = sum(len(v) for v in result.values())
    print(f"Saved {total} videos → {args.output}")

    # Print one example entry
    for phase in ("train", "dev", "test"):
        if result.get(phase):
            vid, entry = next(iter(result[phase].items()))
            fr = entry["Frame_Range"]
            print(f"\nExample [{phase}] {vid}:")
            print(f"  Translation:  {entry['Translation'][:80]}")
            print(f"  Segments:     {len(fr)}")
            print(f"  Seg lengths:  {[len(s) for s in fr]}")
            print(f"  Frame_Range[0] (First Segment): {fr[0]}")
            break


if __name__ == "__main__":
    main()
