"""
Extracts per-token MBart encoder embeddings from translation sentences.

These embeddings are the text side of the CLCL contrastive loss in Stage 1,
and are precomputed offline to avoid re-encoding translations every epoch.

The metadata file (--metadata, output of prepare_metadata.py) provides the
video-to-translation mapping that this script iterates over.

Outputs a .pkl file with structure:
  {
    "train": { video_id -> {
        "token_input_embeddings":  np.ndarray (T, D),   # embedding layer output
        "token_output_embeddings": np.ndarray (T, D),   # encoder hidden states
        "pooled_input_embeddings": np.ndarray (D,),
        "pooled_output_embeddings": np.ndarray (D,),
    }},
    "dev":  { ... },
    "test": { ... },
  }

Usage:
  # Phoenix-2014T
  python scripts/prepare_sentence_embeddings.py phx14t \
    --metadata support_files/metadata/phoenix14T_metadata.pkl \
    --mbart_path pretrain_models/MBart_trimmed \
    --output support_files/phoenix14T_sent_embeddings.pkl

  # CSL-Daily
  python scripts/prepare_sentence_embeddings.py csldaily \
    --metadata support_files/metadata/csldaily_metadata.pkl \
    --mbart_path pretrain_models/MBart_trimmed_csldaily \
    --output support_files/csldaily_sent_embeddings.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBartTokenizer


# MBart appends </s> (EOS=2) + lang_code at the end of every encoder input.
# Strip those; keep all subword content tokens.
# PHX14T trimmed vocab=2172: de_DE=2148; CSLDaily trimmed vocab=6036: zh_CN=6034.
_SPECIAL_TOKEN_IDS_PHX  = {2, 2148}  # </s>, de_DE
_SPECIAL_TOKEN_IDS_CSL  = {2, 6034}  # </s>, zh_CN


def _load_video_meta(path):
    with open(path, "rb") as f:
        meta = pickle.load(f)
    if "video_meta" not in meta:
        raise KeyError(
            f"'video_meta' not found in {path}. "
            "Re-run prepare_metadata.py to regenerate the metadata file."
        )
    return meta["video_meta"]  # {phase: {vid: {"translation": str}}}


def _encode(video_meta, extractor, tokenizer, device,
            special_token_ids, sentence_transform=None, desc="") -> dict:
    """
    Generic per-token embedding extractor.
    Strips EOS and lang-code tokens; keeps all remaining subword tokens.
    """
    result = {}
    for phase in (p for p in ("train", "dev", "test") if p in video_meta):
        phase_emb = {}
        for vid, info in tqdm(video_meta[phase].items(), desc=f"{desc} [{phase}]"):
            sentence = info["translation"]
            if sentence_transform:
                sentence = sentence_transform(sentence)

            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                in_emb = extractor.embed_tokens(inputs["input_ids"])
                out = extractor(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                in_emb = in_emb.squeeze(0)
                out_emb = out.last_hidden_state.squeeze(0)

            kept_in, kept_out = [], []
            for tid, ie, oe in zip(inputs["input_ids"][0], in_emb, out_emb):
                if tid.item() not in special_token_ids:
                    kept_in.append(ie)
                    kept_out.append(oe)

            if not kept_in:
                continue
            kept_in  = torch.stack(kept_in)
            kept_out = torch.stack(kept_out)
            phase_emb[vid] = {
                "token_input_embeddings":  kept_in.cpu().numpy(),
                "token_output_embeddings": kept_out.cpu().numpy(),
                "pooled_input_embeddings":  kept_in.mean(0).cpu().numpy(),
                "pooled_output_embeddings": kept_out.mean(0).cpu().numpy(),
            }
        result[phase] = phase_emb
    return result


def encode_phx14t(video_meta, extractor, tokenizer, device):
    return _encode(video_meta, extractor, tokenizer, device,
                   special_token_ids=_SPECIAL_TOKEN_IDS_PHX, desc="PHX14T")


def encode_csldaily(video_meta, extractor, tokenizer, device):
    return _encode(video_meta, extractor, tokenizer, device,
                   special_token_ids=_SPECIAL_TOKEN_IDS_CSL,
                   sentence_transform=lambda s: s.replace("，", ""),
                   desc="CSLDaily")


_LANG_CODES = {"phx14t": "de_DE", "csldaily": "zh_CN"}
_ENCODERS   = {"phx14t": encode_phx14t, "csldaily": encode_csldaily}


def main():
    parser = argparse.ArgumentParser(description="Extract MBart sentence embeddings for SAGE")
    sub = parser.add_subparsers(dest="dataset", required=True)

    for name in ("phx14t", "csldaily"):
        p = sub.add_parser(name)
        p.add_argument("--metadata", required=True,
                        help="Metadata .pkl from prepare_metadata.py")
        p.add_argument("--mbart_path", required=True,
                        help="Path to trimmed MBart model directory")
        p.add_argument("--output", required=True, help="Output .pkl path")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    lang = _LANG_CODES[args.dataset]
    device = torch.device(args.device)
    print(f"Dataset: {args.dataset} | Lang: {lang} | Device: {device}")

    print("Loading MBart encoder...")
    extractor = (MBartForConditionalGeneration
                 .from_pretrained(args.mbart_path)
                 .get_encoder()
                 .to(device)
                 .eval())
    tokenizer = MBartTokenizer.from_pretrained(args.mbart_path, src_lang=lang, tgt_lang=lang)

    print(f"Loading video metadata from {args.metadata}...")
    video_meta = _load_video_meta(args.metadata)

    result = _ENCODERS[args.dataset](video_meta, extractor, tokenizer, device)

    total = sum(len(v) for v in result.values())
    print(f"Encoded {total} videos across {list(result.keys())} splits")

    for phase, vids in result.items():
        bad = [v for v, embs in vids.items()
               if np.isnan(embs["token_input_embeddings"]).any()
               or np.isnan(embs["token_output_embeddings"]).any()]
        if bad:
            print(f"  WARNING: NaN in {len(bad)} {phase} videos: {bad[:5]}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
