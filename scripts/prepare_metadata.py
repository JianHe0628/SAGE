"""
Extracts pseudo-glosses and video metadata from the label files shipped with SAGE.

The label files (data/Phonexi-2014T/labels.{train,dev,test} and
data/CSL-daily/labels.{train,dev,test}) are gzip-compressed pickles that already
contain video IDs and translations — no external dataset download required.

Outputs a metadata .pkl containing:
  {
    "dict_sentence":    { sentence_str -> [word, ...] },   # pseudo-glosses
    "dict_lem_counter": { word -> count },
    "dict_lem_to_id":   { word -> int },
    "video_meta": {
        "train": { video_id -> {"translation": str} },
        "dev":   { ... },
        "test":  { ... },
    },
  }

Usage:
  # Phoenix-2014T
  python scripts/prepare_metadata.py phx14t \
    --data_dir data/Phonexi-2014T \
    --output support_files/metadata/phoenix14T_metadata.pkl

  # CSL-Daily
  python scripts/prepare_metadata.py csldaily \
    --data_dir data/CSL-daily \
    --output support_files/metadata/csldaily_metadata.pkl
"""

import argparse
import gzip
import pickle
from collections import Counter
from pathlib import Path

import spacy
from tqdm import tqdm


CONTENT_POS = {"NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"}


def _load_label_split(data_dir: Path, split: str) -> dict:
    """Load one gzip-compressed label pkl and return {video_id: {translation: str}}."""
    path = data_dir / f"labels.{split}"
    with gzip.open(path, "rb") as f:
        raw = pickle.load(f)
    result = {}
    for key, entry in raw.items():
        # PHX14T keys are "{phase}/{vid_id}"; CSL keys are just "{vid_id}"
        vid_id = key.split("/", 1)[-1] if "/" in key else key
        result[vid_id] = {"translation": entry["text"]}
    return result


def build_pseudo_gloss(translations: list, nlp, use_lemma: bool = True,
                       exclude_words: set = None) -> dict:
    exclude_words = exclude_words or set()
    dict_sentence = {}
    all_words = []

    for sentence in tqdm(translations, desc="Extracting pseudo-glosses"):
        doc = nlp(sentence)
        words = []
        for token in doc:
            if token.pos_ not in CONTENT_POS:
                continue
            word = token.lemma_.lower() if use_lemma else token.text.lower()
            if word in exclude_words:
                continue
            words.append(word)
        dict_sentence[sentence] = words
        all_words.extend(words)

    return {
        "dict_sentence": dict_sentence,
        "dict_lem_counter": dict(Counter(all_words)),
        "dict_lem_to_id": {w: i for i, w in enumerate(set(all_words))},
    }


def run_phx14t(args):
    import spacy.cli
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        print("Downloading de_core_news_lg...")
        spacy.cli.download("de_core_news_lg")
        nlp = spacy.load("de_core_news_lg")

    data_dir = Path(args.data_dir)
    video_meta = {}
    all_translations = []
    for split in ("train", "dev", "test"):
        video_meta[split] = _load_label_split(data_dir, split)
        all_translations.extend(v["translation"] for v in video_meta[split].values())

    result = build_pseudo_gloss(
        list(dict.fromkeys(all_translations)),
        nlp, use_lemma=True, exclude_words={"es", "sich"}
    )
    result["video_meta"] = video_meta
    return result


def run_csldaily(args):
    import spacy.cli
    try:
        nlp = spacy.load("zh_core_web_sm")
    except OSError:
        print("Downloading zh_core_web_sm...")
        spacy.cli.download("zh_core_web_sm")
        nlp = spacy.load("zh_core_web_sm")

    data_dir = Path(args.data_dir)
    video_meta = {}
    all_translations = []
    for split in ("train", "dev", "test"):
        video_meta[split] = _load_label_split(data_dir, split)
        all_translations.extend(v["translation"] for v in video_meta[split].values())

    result = build_pseudo_gloss(
        list(dict.fromkeys(all_translations)),
        nlp, use_lemma=False
    )
    result["video_meta"] = video_meta
    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare SAGE metadata files")
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_phx = sub.add_parser("phx14t", help="Phoenix-2014T (German)")
    p_phx.add_argument("--data_dir", default="data/Phonexi-2014T",
                        help="Directory containing labels.{train,dev,test} (default: data/Phonexi-2014T)")
    p_phx.add_argument("--output", required=True, help="Output .pkl path")

    p_csl = sub.add_parser("csldaily", help="CSL-Daily (Chinese)")
    p_csl.add_argument("--data_dir", default="data/CSL-daily",
                        help="Directory containing labels.{train,dev,test} (default: data/CSL-daily)")
    p_csl.add_argument("--output", required=True, help="Output .pkl path")

    args = parser.parse_args()

    runners = {"phx14t": run_phx14t, "csldaily": run_csldaily}
    result = runners[args.dataset](args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    n_sentences = len(result["dict_sentence"])
    n_videos = sum(len(v) for v in result["video_meta"].values())
    print(f"Saved {n_sentences} sentences, {n_videos} videos → {args.output}")


if __name__ == "__main__":
    main()
