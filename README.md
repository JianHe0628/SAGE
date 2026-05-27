# SAGE: Segment-Aware Gloss-Free Sign Language Translation

[![ICCVW](https://img.shields.io/badge/ICCVW-2025-blue)](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Low_SAGE_Segment-Aware_Gloss-Free_Encoding_for_Token-Efficient_Sign_Language_Translation_ICCVW_2025_paper.pdf)
[![ArXiv](https://img.shields.io/badge/ArXiv-2507.09266-red)](https://arxiv.org/abs/2507.09266)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)

Official implementation of **"SAGE: Segment-Aware Gloss-Free Sign Language Translation"**, accepted at ICCV Workshop 2025.

## Overview

SAGE is a two-stage gloss-free sign language translation framework that exploits sign segmentation to capture segment-level visual structure without requiring gloss supervision.

**Stage 1 — Cross-Lingual Contrastive Learning (CLCL) Pretraining**
The visual encoder is pretrained by aligning segment-level sign features with pseudo-gloss text embeddings from MBart using a contrastive objective. Pseudo-glosses are automatically extracted from translation sentences via POS tagging, requiring no manual gloss annotation.

**Stage 2 — Translation Fine-tuning**
The pretrained encoder is connected to a full MBart encoder-decoder and fine-tuned end-to-end for sign language translation.

## Installation

```bash
conda activate mae_slt

# Additional packages not in base env
pip install timm
pip install vidaug @ git+https://github.com/okankop/vidaug
```

For the data preparation scripts only (pseudo-gloss extraction):
```bash
pip install spacy
```

> Tested with Python 3.10, PyTorch 2.11, CUDA 12.8.

## Data Preparation

SAGE requires three precomputed support files per dataset:

| File | Config key | Description |
|------|-----------|-------------|
| Metadata | `data.metadata_path` | Pseudo-glosses + video translations (step 1) |
| Segment annotations | `data.segment_path` | Frame-level sign segments (step 2) |
| Sentence embeddings | `data.sentence_embedding_path` | MBart token embeddings per video (step 3) |

An LMDB image store (`data.lmdb_path`) must also be built from the raw video frames.

### 1. Generate metadata

This step extracts pseudo-glosses (POS tagging) and builds the video-to-translation
mapping used by steps 2 and 3. It reads from the label files already included in
`data/` — no external dataset download required for this step.

```bash
# Phoenix-2014T
python scripts/prepare_metadata.py phx14t \
  --output support_files/metadata/phoenix14T_metadata.pkl

# CSL-Daily
python scripts/prepare_metadata.py csldaily \
  --output support_files/metadata/csldaily_metadata.pkl
```

### 2. Generate segment annotations

Raw segmentor outputs are shipped in `support_files/segment_annotations/`
(`phoenix14T_segmentor_outs.pt` and `csldaily_segmentor_outs.pt`). Requires the metadata file from step 1.

```bash
# Phoenix-2014T
python scripts/prepare_segment_annotations.py phx14t \
  --segs support_files/segment_annotations/phoenix14T_segmentor_outs.pt \
  --metadata support_files/metadata/phoenix14T_metadata.pkl \
  --output support_files/phoenix14T_segments.pkl

# CSL-Daily
python scripts/prepare_segment_annotations.py csldaily \
  --segs support_files/segment_annotations/csldaily_segmentor_outs.pt \
  --metadata support_files/metadata/csldaily_metadata.pkl \
  --output support_files/csldaily_segments.pkl
```

### 3. Extract sentence embeddings

```bash
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
```

### 4. Configure paths

Edit `configs/sage.yaml` to point to your dataset and support files:

```yaml
data:
  train_label_path: ./data/Phonexi-2014T/labels.train
  metadata_path:    support_files/metadata/phoenix14T_metadata.pkl
  segment_path:     support_files/phoenix14T_segments.pkl
  sentence_embedding_path: support_files/phoenix14T_sent_embeddings.pkl
  lmdb_path: /path/to/Phoenix14T_Segments_ImgBytes
model:
  tokenizer:   pretrain_models/MBart_trimmed
  transformer: pretrain_models/MBart_trimmed
  visual_encoder: pretrain_models/mytran
```

## Training

Configuration is managed with [Hydra](https://hydra.cc). Stage-specific settings live in `configs/stage1.yaml` and `configs/stage2.yaml`, which both inherit from `configs/sage.yaml`.

### Stage 1: CLCL Pretraining

```bash
python stage1_pretraining.py
```

Override any config value on the command line:

```bash
python stage1_pretraining.py batch_size=16 lr=0.01 output_dir=out/stage1
```

### Stage 2: Translation Fine-tuning

```bash
python stage2_translation.py \
  finetune=out/stage1/SAGE_.../best_checkpoint.pth \
  output_dir=out/stage2
```

### Evaluation only

```bash
python stage2_translation.py \
  resume=out/stage2/SAGE_.../best_checkpoint.pth \
  eval=true
```

## Results

### Phoenix-2014T

| Method | Dev BLEU-4 | Test BLEU-4 |
|--------|-----------|------------|
| SAGE (ours) | — | — |

### CSL-Daily

| Method | Dev BLEU-4 | Test BLEU-4 |
|--------|-----------|------------|
| SAGE (ours) | — | — |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{low2025sage,
  title     = {SAGE: Segment-Aware Gloss-Free Encoding for Token-Efficient Sign Language Translation},
  author    = {Low, Jian He and Sincan, Ozge Mercanoglu and Bowden, Richard},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2025}
}
```

## Acknowledgements

This codebase builds upon [GFSLT-VLP](https://arxiv.org/abs/2307.14768) (Zhou et al., ICCV 2023). We thank the authors for open-sourcing their work.

## License

MIT License
