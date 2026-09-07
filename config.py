"""Case 1 提交包的配置与可复现性工具。

默认值对应论文正文和补充材料中报告的超参数；论文未明确给出的细节也在此
显式写出，以保证提交代码可以确定性复现。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


STAR_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\Users\mikas\Desktop\first_datasets\1-64_exp")
DEFAULT_RUN_DIR = STAR_DIR.parent / "case1_run"


@dataclass
class Case1Config:
    # 数据与训练/测试划分
    data_root: Path = DEFAULT_DATA_ROOT
    output_dir: Path = STAR_DIR
    run_dir: Path = DEFAULT_RUN_DIR
    sequence_length: int = 100
    heatmap_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.70
    seed: int = 42

    # 补充材料给出的模型超参数
    time_input_dim: int = 2
    heatmap_channels: int = 3
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.10

    # 优化器与损失函数
    batch_size: int = 8
    epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    huber_delta: float = 1.0
    lambda_corr: float = 0.05
    lambda_cons: float = 0.10
    jitter_std: float = 0.01

    # 论文未明确给出的实现假设
    branch_channels: int = 64
    tcn_kernel_size: int = 3
    dcn_hidden_channels: int = 64
    gaussian_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("data_root", "output_dir", "run_dir"):
            data[key] = str(data[key])
        data["heatmap_size"] = list(data["heatmap_size"])
        data["gaussian_sigmas"] = list(data["gaussian_sigmas"])
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    """固定数据读取与模型训练使用的全部随机数发生器。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 目标运行环境支持 DCN 的确定性 CPU/CUDA 路径；固定 CuDNN 设置可以
    # 避免不同训练/测试划分运行时出现隐式差异。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=STAR_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu, cuda, or omitted for auto selection")
    return parser


def config_from_args(args: argparse.Namespace) -> Case1Config:
    return Case1Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
