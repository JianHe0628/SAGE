import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
import subprocess
from metrics.Metrics import calc_translation_score

from transformers import MBartTokenizer

from sage.models.model import SAGETranslator
from sage.utils import utils
from sage.utils.training import detect_dataset, make_dataloader, save_checkpoint

import os
import time
import json
import datetime
import numpy as np
from collections import OrderedDict
import random
import wandb
from pathlib import Path
from typing import Iterable
import math
import sys
from loguru import logger

import hydra
from omegaconf import DictConfig, OmegaConf

from sacrebleu.metrics import BLEU

from timm.optim import create_optimizer
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from sage.utils.definition import *


def _run_name(cfg: DictConfig) -> str:
    finetune_tag = f'_ft{Path(cfg.finetune).parent.name}' if cfg.finetune else ''
    return f'SAGE_bs{cfg.batch_size}_opt_{cfg.opt}_lr{cfg.lr}{finetune_tag}_{cfg.comments}'


def log_gpu_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    try:
        gpu_memory = float(result.stdout.strip()) / 1024
    except ValueError:
        gpu_memory = 0.0
    wandb.log({"GPU VRAM (GB)": gpu_memory})


@hydra.main(config_path="configs", config_name="stage2", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_object(cfg)
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cudnn.benchmark = True

    device = torch.device(cfg.device)
    dataset_name, lang = detect_dataset(str(config['data']['train_label_path']))
    print(f"Dataset: {dataset_name} | Language: {lang}")

    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    annot_path = cfg.seg_annotation if cfg.seg_annotation else config['data']['segment_path']
    dev_phase = 'val'

    train_dataloader = make_dataloader('train', annot_path, tokenizer, config, cfg, dataset_name, drop_last=True)
    dev_dataloader   = make_dataloader(dev_phase, annot_path, tokenizer, config, cfg, dataset_name)
    test_dataloader  = make_dataloader('test', annot_path, tokenizer, config, cfg, dataset_name)

    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'], src_lang=lang, tgt_lang=lang)
    amp_dtype = torch.bfloat16 if config['training'].get('bf16', False) else None

    model = SAGETranslator(config, cfg, device=device)
    model.to(device)
    n_parameters = utils.count_parameters_in_MB(model)
    print(f"Parameters: {n_parameters:.1f}M")

    if cfg.finetune:
        print("Loading Visual Encoder weights from stage 1 checkpoint...")
        state_dict = torch.load(cfg.finetune, map_location='cpu', weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if 'conv_2d' in k or 'conv_1d' in k or 'uproject' in k:
                new_state_dict['backbone.' + '.'.join(k.split('.')[2:])] = v
            if 'vis_lang_proj' in k or 'lm_head' in k:
                new_state_dict['.'.join(k.split('.')[1:])] = v
            if 'global_encoder' in k:
                new_state_dict['mbart.model.encoder.' + '.'.join(k.split('.')[2:])] = v

        model_dict = torch.load(config['model']['transformer'] + '/pytorch_model.bin', map_location='cpu', weights_only=False)
        for k, v in model_dict.items():
            if 'decoder.embed_tokens.weight' in k or 'decoder.embed_positions.weight' in k:
                new_state_dict['mbart.' + k] = v

        ret = model.load_state_dict(new_state_dict, strict=False)
        print('Missing keys:\n', '\n'.join(ret.missing_keys))
        print('Unexpected keys:\n', '\n'.join(ret.unexpected_keys))

    optimizer = create_optimizer(cfg, model)
    lr_scheduler = scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=1e-8, T_max=cfg.epochs)

    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0. or cfg.cutmix_minmax is not None
    if mixup_active:
        Mixup(mixup_alpha=cfg.mixup, cutmix_alpha=cfg.cutmix, cutmix_minmax=cfg.cutmix_minmax,
              prob=cfg.mixup_prob, switch_prob=cfg.mixup_switch_prob, mode=cfg.mixup_mode,
              label_smoothing=0.2, num_classes=2454)
    criterion = SoftTargetCrossEntropy() if mixup_active else torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    run_dir = None
    if cfg.output_dir:
        run_dir = Path(cfg.output_dir) / _run_name(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_MODE"] = config['training']['wandb'] if not cfg.eval else 'disabled'
    wandb.login()
    wandb.init(
        project=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_dir.name if run_dir else _run_name(cfg),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("training/*", step_metric="epoch")
    wandb.define_metric("dev/*", step_metric="epoch")

    start_epoch = cfg.start_epoch
    if cfg.resume:
        print('Resuming from checkpoint...')
        checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        if not cfg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    if cfg.eval:
        if not cfg.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(cfg, dev_dataloader, model, tokenizer, criterion, config, lang, device, model_type='dev')
        print(f"Dev BLEU-4: {dev_stats['belu4']:.2f}")
        test_stats = evaluate(cfg, test_dataloader, model, tokenizer, criterion, config, lang, device, model_type='test')
        print(f"Test BLEU-4: {test_stats['belu4']:.2f}")
        return

    print(f"Start training for {cfg.epochs} epochs")
    start_time = time.time()
    max_dev_bleu4 = 0.0
    max_test_bleu4 = 0.0

    for epoch in range(start_epoch, cfg.epochs):
        train_stats = train_one_epoch(cfg, model, criterion, train_dataloader, optimizer, device, epoch, amp_dtype)
        lr_scheduler.step(epoch)

        if run_dir:
            save_checkpoint(run_dir / 'checkpoint.pth', model, optimizer, lr_scheduler, epoch)

        dev_stats = evaluate(cfg, dev_dataloader, model, tokenizer, criterion, config, lang, device)
        print(f"Dev BLEU-4: {dev_stats['belu4']:.2f}")

        actual_test_bleu4 = actual_test_bleu1 = 0.0
        if cfg.log_test and (max_dev_bleu4 < dev_stats['belu4'] or dev_stats['belu4'] > 11):
            test_part_stats = evaluate(cfg, test_dataloader, model, tokenizer, criterion, config, lang, device)
            actual_test_bleu4 = test_part_stats['belu4']
            actual_test_bleu1 = test_part_stats['bleu1']
            if actual_test_bleu4 > max_test_bleu4:
                max_test_bleu4 = actual_test_bleu4
                if run_dir:
                    save_checkpoint(run_dir / 'best_test_checkpoint.pth', model, optimizer, lr_scheduler, epoch, cfg)

        if dev_stats['belu4'] > max_dev_bleu4:
            max_dev_bleu4 = dev_stats['belu4']
            if run_dir:
                save_checkpoint(run_dir / 'best_checkpoint.pth', model, optimizer, lr_scheduler, epoch, cfg)

        print(f"Max Dev BLEU-4: {max_dev_bleu4:.2f}% | Max Test BLEU-4: {max_test_bleu4:.2f}%")
        wandb.log({
            'epoch': epoch + 1,
            'training/train_loss': train_stats['loss'],
            'learning_rate': train_stats['lr'],
            'dev/dev_loss': dev_stats['loss'],
            'dev/Bleu_4': dev_stats['belu4'],
            'dev/Best_Bleu_4': max_dev_bleu4,
            'dev/bleu1': dev_stats['bleu1'],
            'test/bleu4': actual_test_bleu4,
            'test/bleu1': actual_test_bleu1,
        })
        log_gpu_usage()

        if run_dir:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'dev_{k}': v for k, v in dev_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
            }
            with (run_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if run_dir:
        checkpoint = torch.load(run_dir / 'best_checkpoint.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        dev_stats = evaluate(cfg, dev_dataloader, model, tokenizer, criterion, config, lang, device, store_preds=True, model_type='dev')
        print(f"Best checkpoint Dev BLEU-4: {dev_stats['belu4']:.2f}")
        test_stats = evaluate(cfg, test_dataloader, model, tokenizer, criterion, config, lang, device, store_preds=True, model_type='test')
        print(f"Best checkpoint Test BLEU-4: {test_stats['belu4']:.2f}")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(cfg: DictConfig, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, amp_dtype: torch.dtype | None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{cfg.epochs}]'

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            out_logits = model(src_input, tgt_input)
            label = tgt_input['input_ids'].reshape(-1)
            logits = out_logits.reshape(-1, out_logits.shape[-1])
            loss = criterion(logits, label.to(device, non_blocking=True))

        optimizer.zero_grad()
        loss.backward()
        if cfg.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad, norm_type=2)
        optimizer.step()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_mbart=round(float(optimizer.param_groups[1]["lr"]), 8))

        if (step + 1) % 10 == 0 and cfg.visualize:
            utils.visualization(model.visualize())

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(cfg: DictConfig, dev_dataloader, model, tokenizer, criterion, config,
             lang: str, device, store_preds=False, model_type='test'):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    tgt_pres, tgt_refs = [], []

    print(f'Eval Language: {lang}')
    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, 'Eval:')):
            out_logits = model(src_input, tgt_input)
            label = tgt_input['input_ids'].reshape(-1)
            logits = out_logits.reshape(-1, out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))
            metric_logger.update(loss=tgt_loss.item())

            output = model.generate(src_input, max_new_tokens=150, num_beams=4,
                                    decoder_start_token_id=tokenizer.lang_code_to_id[lang])

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i, :])
                tgt_refs.append(tgt_input['input_ids'][i, :])

            if (step + 1) % 10 == 0 and cfg.visualize:
                utils.visualization(model.visualize())

    # Ensure the first sequence is at least 200 tokens so pad_sequence produces a consistent width
    for seq_list in (tgt_pres, tgt_refs):
        pad_len = 200 - len(seq_list[0])
        if pad_len > 0:
            seq_list[0] = torch.cat((seq_list[0], torch.ones(pad_len, device=device).long()), dim=0)
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=PAD_IDX)
    tgt_refs = pad_sequence(tgt_refs, batch_first=True, padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    train_label = str(config['data']['train_label_path'])
    if train_label.endswith('/data/Phonexi-2014T/labels.train'):
        tgt_pres = [s + " ." for s in tgt_pres]
        tgt_refs = [s + " ." for s in tgt_refs]
    elif train_label.endswith('/data/CSL-daily/labels.train'):
        tgt_pres = [' '.join(list(s)) for s in tgt_pres]
        tgt_refs = [' '.join(list(s)) for s in tgt_refs]

    for i in range(min(5, len(tgt_refs))):
        print(f"Ref [{i}]: {tgt_refs[i]}")
        print(f"Hyp [{i}]: {tgt_pres[i]}")

    if store_preds or cfg.eval:
        out_dir = cfg.eval_output if cfg.eval_output else cfg.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{out_dir}/{model_type}_pres.txt', 'w') as f:
            f.write('\n'.join(tgt_pres) + '\n')
        with open(f'{out_dir}/{model_type}_refs.txt', 'w') as f:
            f.write('\n'.join(tgt_refs) + '\n')
        calc_translation_score(out_dir, mode=model_type)

    bleu4_s = BLEU().corpus_score(tgt_pres, [tgt_refs]).score
    metric_logger.meters['belu4'].update(bleu4_s)

    bleu1_s = BLEU(max_ngram_order=1).corpus_score(tgt_pres, [tgt_refs]).score if cfg.log_test else 0.0
    metric_logger.meters['bleu1'].update(bleu1_s)

    metric_logger.synchronize_between_processes()
    print(f'* BLEU-4 {metric_logger.belu4.global_avg:.3f} | Loss {metric_logger.loss.global_avg:.3f}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
