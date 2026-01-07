"""
Stage 1: 高配 RNN 训练脚本（不使用 WFST/Redis），默认“拉满”训练规模。
可在 8×A100 上跑，使用更大的 GRU、更多训练步和更大批次。

示例：
python model_training/hiend_pipeline/stage1_train_rnn.py \
  --gpu 0 \
  --output_dir trained_models/max_rnn \
  --checkpoint_dir trained_models/max_rnn/checkpoint
"""

import argparse
import os
import sys
import pathlib
from omegaconf import OmegaConf

# add repo/model_training to path for direct script execution
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training"))

from rnn_trainer import BrainToTextDecoder_Trainer


def build_args(base_args, output_dir, checkpoint_dir, gpu):
    cfg = base_args.copy()

    # 模型放大 + 更复杂前端（输入多层 MLP + LayerNorm）
    cfg.model.n_units = 1280
    cfg.model.n_layers = 8
    cfg.model.rnn_dropout = 0.35
    cfg.model.patch_size = 18
    cfg.model.patch_stride = 4
    cfg.model.input_network.n_input_layers = 2
    cfg.model.input_network.input_layer_sizes = [768, 512]
    cfg.model.input_network.input_layer_dropout = 0.1

    # 训练规模提升
    cfg.num_training_batches = 300000
    cfg.lr_max = 0.007
    cfg.lr_min = 0.00005
    cfg.lr_decay_steps = cfg.num_training_batches
    cfg.lr_max_day = 0.007
    cfg.lr_min_day = 0.00005
    cfg.lr_decay_steps_day = cfg.num_training_batches
    cfg.lr_warmup_steps = 3000
    cfg.lr_warmup_steps_day = 3000
    cfg.grad_norm_clip_value = 5
    cfg.batches_per_val_step = 2000
    cfg.batches_per_train_log = 200

    # 数据侧加大批次/worker
    cfg.dataset.batch_size = 128
    cfg.dataset.days_per_batch = 8
    cfg.dataset.num_dataloader_workers = 8
    cfg.dataset.loader_shuffle = True

    # 轻微增加增强幅度
    cfg.dataset.data_transforms.white_noise_std = 1.5
    cfg.dataset.data_transforms.constant_offset_std = 0.3
    cfg.dataset.data_transforms.random_cut = 5

    # 训练输出
    cfg.output_dir = output_dir
    cfg.checkpoint_dir = checkpoint_dir
    cfg.gpu_number = str(gpu)
    cfg.mode = "train"
    cfg.save_best_checkpoint = True
    cfg.save_final_model = True

    # 设定种子
    cfg.seed = 42
    cfg.dataset.seed = 42

    return cfg


def main():
    parser = argparse.ArgumentParser(description="High-capacity RNN training (stage 1)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="trained_models/max_rnn",
        help="Where to save logs/metrics",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="trained_models/max_rnn/checkpoint",
        help="Where to save checkpoints",
    )
    args = parser.parse_args()

    base_args = OmegaConf.load(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rnn_args.yaml"))
    )

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", args.output_dir))
    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", args.checkpoint_dir)
    )
    # 如果目录已存在，自动追加 _runX 后缀，避免 Trainer 内部的 exist_ok=False 报错
    base_out = output_dir
    base_ckpt = checkpoint_dir
    run_idx = 0
    while os.path.exists(output_dir):
        run_idx += 1
        output_dir = f"{base_out}_run{run_idx}"
        checkpoint_dir = f"{base_ckpt}_run{run_idx}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg = build_args(base_args, output_dir, checkpoint_dir, args.gpu)

    # 保存使用的配置
    OmegaConf.save(cfg, os.path.join(output_dir, "args_used.yaml"))

    trainer = BrainToTextDecoder_Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
