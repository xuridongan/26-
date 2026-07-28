"""
PP-YOLOE+ S 安全城市场景目标检测训练脚本
========================================
用法:
  python train.py                              # GPU 训练（推荐）
  python train.py --epochs 1 --batch_size 2    # 快速测试
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import paddle
import yaml

from ppdet.core.workspace import global_config, merge_config, create, load_config
from ppdet.engine import Trainer
from ppdet.engine.callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer
from ppdet.utils.checkpoint import load_pretrain_weight, load_weight
from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
CKPT_DIR = OUTPUT_DIR / "checkpoints"

CLASS_NAMES = [
    "collapsed building", "building on fire",
    "building with toxic gas", "building with a power failure",
    "medical rescue groups", "general rescue groups",
    "kitchen waste bin", "recyclable waste bin",
    "hazardous waste bin", "residual waste bin",
]
NUM_CLASSES = len(CLASS_NAMES)


def write_config_yaml(args):
    """Write a self-contained PP-YOLOE+ S config YAML."""
    cfg = {
        'architecture': 'PPYOLOE',
        'pretrain_weights': (
            'https://paddledet.bj.bcebos.com/models/'
            'ppyoloe_plus_crn_s_60e_objects365.pdparams'
        ),
        'weights': str(OUTPUT_DIR / 'model_final.pdparams'),
        'norm_type': 'bn',
        'use_ema': True,
        'ema_decay': 0.9998,
        'epochs': args.epochs,
        'epoch': args.epochs,
        'snapshot_epoch': 10,
        'log_iter': 10,
        'eval_epoch_interval': 10,
        'num_classes': NUM_CLASSES,
        'save_dir': str(OUTPUT_DIR),
        'metric': 'COCO',

        # Backbone (PP-YOLOE+ S: depth×0.33, width×0.50)
        'backbone': 'CSPResNet',
        'CSPResNet': {
            'act': 'swish',
            'depth_mult': 0.33, 'width_mult': 0.50,
            'return_idx': [1, 2, 3],
            'use_large_stem': True,
        },

        # Neck
        'neck': 'CustomCSPPAN',
        'CustomCSPPAN': {
            'act': 'swish',
            'depth_mult': 0.33, 'width_mult': 0.50,
        },

        # Head
        'yolo_head': 'PPYOLOEHead',
        'PPYOLOEHead': {
            'num_classes': NUM_CLASSES,
            'act': 'swish',
            'fpn_strides': [32, 16, 8],
            'grid_cell_scale': 5.0, 'grid_cell_offset': 0.5, 'reg_max': 16,
            'static_assigner_epoch': 4,
            'use_varifocal_loss': True,
            'loss_weight': {
                'class': args.loss_cls_weight,
                'iou': 2.5,
                'dfl': 0.5,
            },
            'static_assigner': 'ATSSAssigner',
            'assigner': 'TaskAlignedAssigner',
            'nms': 'MultiClassNMS',
        },
        'ATSSAssigner': {'topk': 9},
        'TaskAlignedAssigner': {'topk': 13, 'alpha': 1.0, 'beta': 6.0},
        'MultiClassNMS': {
            'score_threshold': 0.01, 'nms_top_k': 1000,
            'keep_top_k': 100, 'nms_threshold': 0.7,
        },

        'PPYOLOEDecode': {'reg_max': 16},

        # Train dataset — 增强后的数据
        'TrainDataset': {
            'name': 'COCODataSet',
            'dataset_dir': str(DATA_DIR / 'augmented'),
            'image_dir': 'image',
            'anno_path': 'train.json',
            'data_fields': ['image', 'gt_bbox', 'gt_class', 'gt_score'],
            'allow_empty': True,
        },

        # Eval dataset — 原始 20 张图
        'EvalDataset': {
            'name': 'COCODataSet',
            'dataset_dir': str(DATA_DIR),
            'image_dir': 'image',
            'anno_path': 'augmented/val.json',
            'allow_empty': True,
        },

        # Optimizer
        'optimizer': 'AdamW',
        'AdamW': {'weight_decay': 5e-4},
        'LearningRate': {
            'base_lr': args.lr,
            'schedulers': [
                {'name': 'CosineDecay', 'max_epochs': args.epochs},
                {'name': 'LinearWarmup', 'steps': 100, 'start_factor': 0.1},
            ],
        },
        'OptimizerBuilder': {
            'clip_grad_by_norm': None,
        },

        # TrainReader（温和增强 + PadGT 生成 pad_gt_mask）
        'TrainReader': {
            'sample_transforms': [
                {'Decode': {}},
                {'RandomDistort': {
                    'hue': [-6, 6, 0.5],
                    'saturation': [0.7, 1.3, 0.5],
                    'contrast': [0.7, 1.3, 0.5],
                    'brightness': [0.9, 1.1, 0.5],
                    'random_apply': True, 'count': 4, 'prob': 0.8,
                }},
                {'RandomFlip': {'prob': 0.5}},
                {'Resize': {
                    'target_size': [640, 640],
                    'keep_ratio': False, 'interp': 2,
                }},
            ],
            'batch_transforms': [
                {'NormalizeImage': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'is_scale': True, 'norm_type': 'mean_std',
                }},
                {'Permute': {}},
                {'PadGT': {}},
            ],
            'batch_size': args.batch_size,
            'shuffle': True, 'drop_last': True,
            'num_workers': 4,
            'use_shared_memory': False,
            'collate_batch': True,
        },

        # EvalReader
        'EvalReader': {
            'sample_transforms': [
                {'Decode': {}},
                {'Resize': {
                    'target_size': [640, 640],
                    'keep_ratio': False, 'interp': 2,
                }},
                {'NormalizeImage': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'is_scale': True, 'norm_type': 'mean_std',
                }},
                {'Permute': {}},
            ],
            'batch_size': args.batch_size,
            'shuffle': False, 'drop_last': False,
            'num_workers': 1,
        },

        # EvalMetric
        'EvalMetric': {
            'name': 'COCOMetric',
            'anno_file': str(DATA_DIR / 'augmented' / 'val.json'),
        },

        # Workers
        'worker_num': 2,

        # Device
        'use_gpu': paddle.is_compiled_with_cuda(),
        'amp': True,
        'amp_cfg': {
            'use_amp': True,
            'enable_amp': True,
            'scale_loss': 128.0,
            'use_dynamic_loss_scaling': True,
        },
    }

    yml_path = PROJECT_DIR / 'config.yaml'
    with open(yml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=None, width=120)
    logger.info(f"Config written: {yml_path}")
    return yml_path


class FreezeBackboneCallback(Callback):
    """Epoch-level backbone freeze/unfreeze."""
    def __init__(self, trainer, freeze_epochs):
        super().__init__(trainer)
        self.freeze_epochs = freeze_epochs

    def on_epoch_begin(self, status):
        epoch = status['epoch_id']
        if self.freeze_epochs <= 0:
            return
        backbone = getattr(self.model.model, 'backbone', None)
        if backbone is None:
            return
        frozen = epoch < self.freeze_epochs
        for p in backbone.parameters():
            p.stop_gradient = frozen
        if epoch == 0 or epoch == self.freeze_epochs:
            logger.info(f"Backbone {'frozen' if frozen else 'UNFROZEN'} (epoch {epoch})")


class EarlyStopCallback(Callback):
    """早停: 监控 loss, 连续 patience 轮不下降则终止训练。"""
    def __init__(self, trainer, patience, min_delta=0.001):
        super().__init__(trainer)
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.best_epoch = -1

    def on_epoch_end(self, status):
        # training_staus 中记录了 loss 信息
        loss = status.get('training_staus', None)
        if loss is None:
            return
        # 取平均 loss
        try:
            avg_loss = loss.get_stats()['loss'].avg
        except:
            return

        if avg_loss < self.best - self.min_delta:
            self.best = avg_loss
            self.best_epoch = status['epoch_id']
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"EarlyStopping @ epoch {status['epoch_id']} "
                    f"(best: {self.best:.4f} @ epoch {self.best_epoch})"
                )
                # 修改 cfg.epoch 使训练循环终止
                self.model.cfg.epoch = status['epoch_id'] + 1


def main():
    parser = argparse.ArgumentParser(description="PP-YOLOE+ S 训练")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--freeze_backbone', type=int, default=30)
    parser.add_argument('--loss_cls_weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='本地预训练权重路径，代替 COCO 预训练')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # 1. 写 YAML 配置
    yml_path = write_config_yaml(args)
    cfg = load_config(str(yml_path))

    # 2. 创建 Trainer
    trainer = Trainer(cfg, mode='train')
    # 禁用外部 BBoxPostProcess（当前版本 PPYOLOEHead 无 mask_anchors）
    trainer.model.post_process = None

    # 3. 加载权重
    if args.resume:
        resume_path = str(args.resume).rstrip(".pdparams").rstrip(".pdopt")
        ckpt_prefix = str(PROJECT_DIR / resume_path)
        ckpt_file = ckpt_prefix + ".pdparams"
        pdopt_file = ckpt_prefix + ".pdopt"
        logger.info(f"[RESUME] args.resume = '{args.resume}'")
        logger.info(f"[RESUME] PROJECT_DIR = {PROJECT_DIR}")
        logger.info(f"[RESUME] ckpt_prefix  = {ckpt_prefix}")
        logger.info(f"[RESUME] .pdparams exists: {Path(ckpt_file).exists()}  (os: {os.path.exists(ckpt_file)})")
        logger.info(f"[RESUME] .pdopt  exists: {Path(pdopt_file).exists()}  (os: {os.path.exists(pdopt_file)})")

        if os.path.exists(ckpt_file) and os.path.exists(pdopt_file):
            # Read last_epoch from pdopt directly as a cross-check
            try:
                pdopt_data = paddle.load(pdopt_file)
                actual_last_epoch = pdopt_data.get('last_epoch', 0)
                logger.info(f"[RESUME] .pdopt last_epoch = {actual_last_epoch}")
            except Exception as e:
                logger.warning(f"[RESUME] Could not read pdopt: {e}")
                actual_last_epoch = 0

            logger.info(f"[RESUME] Calling trainer.resume_weights(...)")
            trainer.resume_weights(ckpt_prefix)
            logger.info(f"[RESUME] After resume_weights, trainer.start_epoch = {trainer.start_epoch}")

            # Backup: if resume_weights didn't set start_epoch correctly, force it
            if trainer.start_epoch == 0 and actual_last_epoch > 0:
                logger.warning(f"[RESUME] resume_weights returned 0, forcing start_epoch = {actual_last_epoch}")
                trainer.start_epoch = actual_last_epoch
        else:
            logger.warning(f"[RESUME] Checkpoint not found, training from Objects365 pretrained.")
            load_pretrain_weight(
                trainer.model,
                'https://paddledet.bj.bcebos.com/models/'
                'ppyoloe_plus_crn_s_60e_objects365.pdparams',
            )
    elif args.pretrained:
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.exists():
            pretrained_path = Path(str(args.pretrained) + '.pdparams')
        if pretrained_path.exists():
            logger.info(f"Loading local pretrained weights: {pretrained_path}")
            load_pretrain_weight(trainer.model, str(pretrained_path))
            logger.info("OK, local pretrained loaded!")
        else:
            logger.warning(f"Pretrained file not found: {pretrained_path}, fallback to Objects365.")
            load_pretrain_weight(
                trainer.model,
                'https://paddledet.bj.bcebos.com/models/'
                'ppyoloe_plus_crn_s_60e_objects365.pdparams',
            )
    else:
        logger.info("Loading Objects365 pretrained weights ...")
        load_pretrain_weight(
            trainer.model,
            'https://paddledet.bj.bcebos.com/models/'
            'ppyoloe_plus_crn_s_60e_objects365.pdparams',
        )
        logger.info("OK, Objects365 pretrained loaded!")

    # 4. 添加自定义 callback
    freeze_cb = FreezeBackboneCallback(trainer, args.freeze_backbone)
    early_cb = EarlyStopCallback(trainer, args.patience)
    log_cb = LogPrinter(trainer)
    ckpt_cb = Checkpointer(trainer)
    trainer._compose_callback = ComposeCallback([freeze_cb, early_cb, log_cb, ckpt_cb])

    # 5. 训练
    logger.info("=" * 60)
    logger.info(f"PP-YOLOE+ S | Classes: {NUM_CLASSES}")
    logger.info(f"Batch: {args.batch_size} | Epochs: {args.epochs}")
    logger.info(f"Freeze backbone: {args.freeze_backbone} | VFL weight: {args.loss_cls_weight}")
    logger.info(f"EarlyStopping patience: {args.patience}")
    logger.info("=" * 60)

    trainer.train(validate=True)

    # 6. 保存最终模型
    paddle.save(trainer.model.state_dict(), str(OUTPUT_DIR / 'model_final.pdparams'))
    logger.info(f"Done! Model saved to {OUTPUT_DIR / 'model_final.pdparams'}")


if __name__ == "__main__":
    main()
