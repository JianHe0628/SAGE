import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from transformers import MBartTokenizer

from sage.models.model import SAGEEncoder
from sage.utils import utils
from sage.utils.training import detect_dataset, make_dataloader, save_checkpoint
from sage.losses.loss import CLCLLoss

import os
import time
import json
import datetime
import numpy as np
from tqdm import tqdm
import random
import wandb
import pickle
from pathlib import Path
from typing import Iterable, Optional
import math
import sys
from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from PIL import Image

from sage.utils.definition import *


def _run_name(cfg: DictConfig) -> str:
    return f'SAGE_bs{cfg.batch_size}_lr{cfg.lr}_tdlr{cfg.td_lr}_lambda{cfg.loss_lambda}_inloss{cfg.inloss_weight}_{cfg.comments}'


def setup_run(cfg: DictConfig):
    os.environ["WANDB_MODE"] = cfg.training.wandb if not cfg.eval else 'disabled'
    wandb.login()
    wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=_run_name(cfg),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("training/*", step_metric="epoch")
    wandb.define_metric("dev/*", step_metric="epoch")


@hydra.main(config_path="configs", config_name="stage1", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_object(cfg)
    print(OmegaConf.to_yaml(cfg))
    setup_run(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cudnn.benchmark = True

    device = torch.device(cfg.device)
    dataset_name, _ = detect_dataset(str(config['data']['train_label_path']))
    print(f"Dataset: {dataset_name}")

    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    annot_path = cfg.seg_annotation if cfg.seg_annotation else config['data']['segment_path']
    dev_phase = 'val'

    train_dataloader = make_dataloader('train', annot_path, tokenizer, config, cfg, dataset_name, drop_last=False)
    dev_dataloader   = make_dataloader(dev_phase, annot_path, tokenizer, config, cfg, dataset_name)
    test_dataloader  = make_dataloader('test', annot_path, tokenizer, config, cfg, dataset_name)

    model = SAGEEncoder(config=config, device=device)
    model.to(device)
    n_parameters = utils.count_parameters_in_MB(model)
    print(f"Parameters: {n_parameters:.1f}M")

    if cfg.finetune:
        checkpoint = torch.load(cfg.finetune, map_location='cpu', weights_only=False)
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys:\n', '\n'.join(ret.missing_keys))
        print('Unexpected keys:\n', '\n'.join(ret.unexpected_keys))

    optimizer = create_optimizer(cfg, model)
    lr_scheduler, _ = create_scheduler(cfg, optimizer)
    criterion = CLCLLoss()
    amp_dtype = torch.bfloat16 if config['training'].get('bf16', False) else torch.float16

    output_dir = None
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir) / _run_name(cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = cfg.start_epoch
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    if cfg.eval:
        if not cfg.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(cfg, dev_dataloader, model, criterion, config, output_dir, epoch=start_epoch)
        print(f"Dev loss: {dev_stats['loss']:.3f}")
        test_stats = evaluate(cfg, test_dataloader, model, criterion, config, output_dir, epoch=start_epoch)
        print(f"Test loss: {test_stats['loss']:.3f}")
        return

    if cfg.ex_unmatched:
        if not cfg.resume:
            raise ValueError('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        if not cfg.output_dir:
            raise ValueError('Please specify the output directory: --output_dir /path/to/output')
        print("Extracting unmatched text features")
        extract_unmatched(cfg, dev_dataloader, model, 'dev')
        extract_unmatched(cfg, test_dataloader, model, 'test')
        extract_unmatched(cfg, train_dataloader, model, 'train')
        return

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    last_epoch = start_epoch

    for epoch in range(start_epoch, cfg.epochs):
        last_epoch = epoch
        train_stats = train_one_epoch(cfg, model, criterion, train_dataloader, optimizer, device, epoch, amp_dtype)
        lr_scheduler.step(epoch)

        if output_dir:
            save_checkpoint(output_dir / 'checkpoint.pth', model, optimizer, lr_scheduler, epoch)

        dev_stats = evaluate(cfg, dev_dataloader, model, criterion, config, output_dir, epoch)
        if dev_stats['loss'] < min_loss:
            min_loss = dev_stats['loss']
            if output_dir:
                save_checkpoint(output_dir / 'best_checkpoint.pth', model, optimizer, lr_scheduler, epoch)

        print(f"* DEV loss {dev_stats['loss']:.3f} | Min {min_loss:.3f}")
        wandb.log({
            'epoch': epoch + 1,
            'training/train_loss': train_stats['loss'],
            'dev/dev_loss': dev_stats['loss'],
            'dev/min_loss': min_loss,
        })

        if output_dir:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'dev_{k}': v for k, v in dev_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
            }
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if output_dir:
        checkpoint = torch.load(output_dir / 'best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        dev_stats = evaluate(cfg, dev_dataloader, model, criterion, config, output_dir, last_epoch)
        print(f"Best checkpoint dev loss: {dev_stats['loss']:.3f}")
        test_stats = evaluate(cfg, test_dataloader, model, criterion, config, output_dir, last_epoch)
        print(f"Best checkpoint test loss: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(cfg: DictConfig, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, amp_dtype: torch.dtype):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{cfg.epochs}]'

    for src_input, tgt_input in metric_logger.log_every(data_loader, 10, header):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            I2T_sim_input, T2I_sim_input, I2T_sim_output, T2I_sim_output, logits = model(src_input, tgt_input)
            in_loss  = (criterion(I2T_sim_input) + criterion(T2I_sim_input.T)) * 0.5
            out_loss = (criterion(I2T_sim_output) + criterion(T2I_sim_output.T)) * 0.5
            total_loss = in_loss * cfg.inloss_weight + out_loss * (1 - cfg.inloss_weight)

        total_loss.backward()
        if cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(f"I2T_sim_input {I2T_sim_input.shape}: {I2T_sim_input}")
            print(f"Image feat: {logits['image_feat']}, Text feat: {logits['text_feat']}")
            sys.exit(1)

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(cfg: DictConfig, dev_dataloader, model, criterion, config,
             output_dir: Optional[Path], epoch: int):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    image_feat_list, text_feat_list = [], []
    image_mask_list, text_mask_list = [], []
    name_batch_list, pgloss_list = [], []

    with torch.no_grad():
        for src_input, tgt_input in metric_logger.log_every(dev_dataloader, 10, 'Eval:'):
            I2T_sim_input, T2I_sim_input, I2T_sim_output, T2I_sim_output, logits = model(src_input, tgt_input)

            if len(image_feat_list) < cfg.num_heatmaps:
                for x in range(logits['image_feat'].shape[0]):
                    image_feat_list.append(logits['image_feat'][x])
                    text_feat_list.append(logits['text_feat'][x])
                    image_mask_list.append(logits['final_image_mask'][x])
                    text_mask_list.append(src_input['text_padding_mask'][x])
                    name_batch_list.append(src_input['name_batch'][x])
                    pgloss_list.append(src_input['batch_pgloss'][x])

            in_loss  = (criterion(I2T_sim_input) + criterion(T2I_sim_input.T)) * 0.5
            out_loss = (criterion(I2T_sim_output) + criterion(T2I_sim_output.T)) * 0.5
            total_loss = in_loss * cfg.inloss_weight + out_loss * (1 - cfg.inloss_weight)
            metric_logger.update(loss=total_loss.item())

    if output_dir and (epoch % 5 == 0 or cfg.eval):
        _save_heatmaps(cfg, config, output_dir,
                       image_feat_list, text_feat_list,
                       image_mask_list, text_mask_list,
                       name_batch_list, pgloss_list, epoch)

    metric_logger.synchronize_between_processes()
    print(f"* DEV loss {metric_logger.loss.global_avg:.3f}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _save_heatmaps(cfg, config, vis_dir: Path,
                   image_feat_list, text_feat_list,
                   image_mask_list, text_mask_list,
                   name_batch_list, pgloss_list, epoch: int):
    train_label = str(config['data']['train_label_path'])
    heatmap_path = None
    n = min(cfg.num_heatmaps, len(image_feat_list))

    for i in range(n):
        img_feat = image_feat_list[i][:image_mask_list[i].sum()]
        txt_feat = text_feat_list[i][:text_mask_list[i].sum()]
        similarity = torch.softmax(torch.matmul(img_feat, txt_feat.T), dim=-1)

        if train_label.endswith('/data/CSL-daily/labels.train'):
            heatmap_path = utils.plot_similarity_heatmap(
                similarity, f'Test {name_batch_list[i]} {pgloss_list[i]}', vis_dir, count=i)
        else:
            heatmap_path = utils.plot_similarity_heatmap(
                similarity, f'Test {name_batch_list[i]}', vis_dir, pgloss_list[i], count=i)

    if n > 1 and heatmap_path is not None:
        fig, axs = plt.subplots(1, n, figsize=(10 * n, 10))
        for i in range(n):
            img = Image.open(vis_dir / f'Similarity_Heatmap_{i}.png')
            axs[i].imshow(img)
            axs[i].set_title(f'Heatmap {i + 1}')
            axs[i].axis('off')
        plt.tight_layout()
        heatmap_path = vis_dir / f'heatmap_combined_epoch_{epoch}.png'
        plt.savefig(heatmap_path)
        plt.close(fig)

    if heatmap_path:
        wandb.log({'Similarity Heatmap': wandb.Image(str(heatmap_path))})


def extract_unmatched(cfg: DictConfig, dataloader, model, split: str):
    model.eval()
    extraction_dict = {}
    with torch.no_grad():
        for src_input, tgt_input in tqdm(dataloader, desc=f'Extracting unmatched [{split}]'):
            _, _, _, _, logits = model(src_input, tgt_input)
            for x in range(logits['image_feat'].shape[0]):
                img_mask = logits['final_image_mask'][x]
                img_feat = logits['image_feat'][x][:img_mask.sum()]
                txt_mask = src_input['text_padding_mask'][x]
                txt_feat = logits['text_feat'][x][:txt_mask.sum()]

                similarity = torch.softmax(torch.matmul(img_feat, txt_feat.T), dim=-1)
                unmatched = torch.where(similarity.sum(dim=0) < 0.6)[0]
                extraction_dict[src_input['name_batch'][x]] = {
                    'image_feat': img_feat.cpu(),
                    'unmatched_text_feat': txt_feat[unmatched].cpu(),
                }

    output_path = Path(cfg.output_dir) / 'Extracted_Unmatched_Text' / f'{split}_unmatched_text_features.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(extraction_dict, f)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
