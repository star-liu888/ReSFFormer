"""训练与论文对应的 Case 1 模型，并将 checkpoint 写到 Star 目录外。"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import Case1Config, config_from_args, make_parser, set_seed
from data import Case1Dataset, Normalizer, load_case1_samples, split_case1_samples
from model import build_model


def correlation_loss(features: Tensor, target: Tensor) -> Tensor:
    """式（18）的负 Fisher-z 变换 Pearson 相关性损失。"""
    if features.shape[0] < 2:
        return features.new_zeros(())
    centered_features = features - features.mean(dim=0, keepdim=True)
    centered_target = target - target.mean()
    numerator = (centered_features * centered_target[:, None]).sum(dim=0)
    denominator = torch.sqrt(
        centered_features.square().sum(dim=0).clamp_min(1e-8)
        * centered_target.square().sum().clamp_min(1e-8)
    )
    rho = (numerator / denominator).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
    return -torch.atanh(rho).mean()


def add_amplitude_jitter(values: Tensor, std: float, generator: torch.Generator | None = None) -> Tensor:
    noise = torch.randn(values.shape, device=values.device, dtype=values.dtype, generator=generator) * std
    return values * (1.0 + noise)


def loss_components(
    model,
    time_series: Tensor,
    heatmap: Tensor,
    target: Tensor,
    config: Case1Config,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    prediction, intermediate = model(time_series, heatmap, return_intermediates=True)
    huber = F.huber_loss(prediction, target, delta=config.huber_delta)
    corr = correlation_loss(intermediate["f_fuse"], target)
    jittered = add_amplitude_jitter(time_series, config.jitter_std)
    jittered_prediction = model(jittered, heatmap)
    consistency = F.mse_loss(prediction, jittered_prediction)
    total = huber + config.lambda_corr * corr + config.lambda_cons * consistency
    return total, {"total": total, "huber": huber, "corr": corr, "consistency": consistency}


def _loader(dataset: Case1Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for time_series, heatmap, target, _ in loader:
            prediction = model(time_series.to(device), heatmap.to(device))
            predictions.append(prediction.cpu())
            targets.append(target)
    y_hat = torch.cat(predictions).numpy()
    y = torch.cat(targets).numpy()
    return {
        "rmse_normalized": float(np.sqrt(np.mean((y_hat - y) ** 2))),
        "mae_normalized": float(np.mean(np.abs(y_hat - y))),
    }


def _full_training_objective(
    model, loader: DataLoader, device: torch.device, config: Case1Config, epoch: int
) -> Dict[str, float]:
    """在全部训练电池上计算式（20），用于选择 checkpoint。

    batch size 为 8 时，单个 mini-batch 的 Pearson 值波动较大。训练仍严格采用
    论文报告的 mini-batch 设置，但 checkpoint 选择不能依赖偶然的单批次相关性。
    此处只使用 45 个训练电池，不会读取测试集结果。
    """
    model.eval()
    predictions, targets, fused_features, jittered_predictions = [], [], [], []
    # 只在本次评估中固定一致性扰动，避免 checkpoint 选择受到随机扰动影响。
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(config.seed + 10_000 + epoch)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed + 10_000 + epoch)
        with torch.no_grad():
            for time_series, heatmap, target, _ in loader:
                time_series = time_series.to(device, non_blocking=True)
                heatmap = heatmap.to(device, non_blocking=True)
                prediction, intermediate = model(time_series, heatmap, return_intermediates=True)
                jittered_prediction = model(add_amplitude_jitter(time_series, config.jitter_std), heatmap)
                predictions.append(prediction)
                targets.append(target.to(device, non_blocking=True))
                fused_features.append(intermediate["f_fuse"])
                jittered_predictions.append(jittered_prediction)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    fused = torch.cat(fused_features)
    jittered_prediction = torch.cat(jittered_predictions)
    huber = F.huber_loss(prediction, target, delta=config.huber_delta)
    corr = correlation_loss(fused, target)
    consistency = F.mse_loss(prediction, jittered_prediction)
    total = huber + config.lambda_corr * corr + config.lambda_cons * consistency
    return {
        "selection_total": float(total.cpu()),
        "selection_huber": float(huber.cpu()),
        "selection_corr": float(corr.cpu()),
        "selection_consistency": float(consistency.cpu()),
    }


def _save_split(path: Path, train, test) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "split"])
        for sample in train:
            writer.writerow([sample.sample_id, "train"])
        for sample in test:
            writer.writerow([sample.sample_id, "test"])


def _scheduler(optimizer: Adam, config: Case1Config) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return float(epoch + 1) / max(1, config.warmup_epochs)
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda=multiplier)


def train(config: Case1Config, device_name: str | None = None) -> Path:
    set_seed(config.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.run_dir / "config.json")

    samples = load_case1_samples(
        config.data_root,
        sequence_length=config.sequence_length,
        heatmap_size=config.heatmap_size,
        expected_count=64,
    )
    train_samples, test_samples = split_case1_samples(samples, config.seed, config.train_ratio)
    normalizer = Normalizer.fit(train_samples)
    (config.run_dir / "normalization.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8"
    )
    _save_split(config.run_dir / "split_manifest.csv", train_samples, test_samples)

    train_dataset = Case1Dataset(train_samples, normalizer)
    test_dataset = Case1Dataset(test_samples, normalizer)
    train_loader = _loader(train_dataset, config.batch_size, True, config.seed)
    test_loader = _loader(test_dataset, config.batch_size, False, config.seed)

    model = build_model(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = _scheduler(optimizer, config)
    best_training_objective = float("inf")
    best_path = config.run_dir / "best.pt"
    history = []

    print(f"Training on {device}; train={len(train_dataset)}, test={len(test_dataset)}")
    for epoch in range(config.epochs):
        model.train()
        sums = {"total": 0.0, "huber": 0.0, "corr": 0.0, "consistency": 0.0}
        count = 0
        for time_series, heatmap, target, _ in train_loader:
            time_series = time_series.to(device, non_blocking=True)
            heatmap = heatmap.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, components = loss_components(model, time_series, heatmap, target, config)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {components}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_size = int(target.shape[0])
            count += batch_size
            for key, value in components.items():
                sums[key] += float(value.detach().cpu()) * batch_size
        scheduler.step()
        train_metrics = {key: value / count for key, value in sums.items()}
        test_metrics = _evaluate(model, test_loader, device)
        selection_metrics = _full_training_objective(model, train_loader, device, config, epoch)
        record = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
            **selection_metrics,
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        history.append(record)
        if selection_metrics["selection_total"] < best_training_objective:
            best_training_objective = selection_metrics["selection_total"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "training_objective": best_training_objective,
                    "epoch": epoch + 1,
                },
                best_path,
            )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == config.epochs:
            print(
                f"epoch {epoch + 1:03d}/{config.epochs} "
                f"train={train_metrics['total']:.5f} "
                f"select={selection_metrics['selection_total']:.5f} "
                f"test_rmse_norm={test_metrics['rmse_normalized']:.5f}"
            )

    (config.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint (selected by training objective): {best_path}")
    return best_path


def main() -> None:
    parser = make_parser("Train the Case 1 AIO-FE + ReSFFormer model")
    args = parser.parse_args()
    config = config_from_args(args)
    train(config, args.device)


if __name__ == "__main__":
    main()
