import torch
from torch.utils.data import DataLoader
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from sage.datasets.dataset import SignDataset


_DATASET_META = {
    '/data/Phonexi-2014T/labels.train': ('PHX', 'de_DE'),
    '/data/CSL-daily/labels.train': ('CSLDaily', 'zh_CN'),
}


def detect_dataset(train_label_path: str) -> tuple:
    """Returns (dataset_name, mbart_lang_code). Raises ValueError for unknown paths."""
    for suffix, meta in _DATASET_META.items():
        if train_label_path.endswith(suffix):
            return meta
    raise ValueError(f"Unknown dataset path: {train_label_path}")


def make_dataloader(phase: str, annot_path, tokenizer, config, cfg: DictConfig,
                    dataset_name: str, **kwargs) -> DataLoader:
    dataset = SignDataset(path=annot_path, tokenizer=tokenizer, config=config, args=cfg,
                          phase=phase, dataset_name=dataset_name)
    print(dataset)
    sampler = torch.utils.data.RandomSampler(dataset)
    return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                      collate_fn=dataset.collate_fn, sampler=sampler,
                      pin_memory=cfg.pin_mem, **kwargs)


def save_checkpoint(path: Path, model, optimizer, lr_scheduler, epoch, cfg: DictConfig = None):
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
    }
    if cfg is not None:
        payload['args'] = OmegaConf.to_container(cfg)
    torch.save(payload, path)
