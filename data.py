"""Case 1 电池数据读取、划分与归一化。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ("工步类型", "电压(V)", "电流(mA)")
CHARGE_STEP = "恒流恒压充电"


@dataclass
class BatterySample:
    sample_id: str
    source_name: str
    life_cycles: float
    time_series: np.ndarray  # 形状为 (T, 2)，依次为电压和电流
    heatmap: np.ndarray  # 形状为 (3, H, W)，RGB 数值范围为 [0, 1]


@dataclass
class Normalizer:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    life_min: float
    life_max: float

    @classmethod
    def fit(cls, samples: Sequence[BatterySample]) -> "Normalizer":
        if not samples:
            raise ValueError("Cannot fit normalisation on an empty training set")
        values = np.concatenate([sample.time_series for sample in samples], axis=0)
        feature_mean = values.mean(axis=0).astype(np.float32)
        feature_std = values.std(axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std).astype(np.float32)
        lives = np.asarray([sample.life_cycles for sample in samples], dtype=np.float32)
        if float(lives.max() - lives.min()) < 1e-8:
            raise ValueError("Training Life range is zero; min-max scaling is undefined")
        return cls(feature_mean, feature_std, float(lives.min()), float(lives.max()))

    def normalize_time(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.feature_mean) / self.feature_std).astype(np.float32)

    def normalize_life(self, value: float) -> float:
        return (float(value) - self.life_min) / (self.life_max - self.life_min)

    def denormalize_life(self, value: np.ndarray | float) -> np.ndarray:
        return np.asarray(value) * (self.life_max - self.life_min) + self.life_min

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "life_min": self.life_min,
            "life_max": self.life_max,
            "feature_method": "training-set z-score",
            "life_method": "training-set min-max",
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "Normalizer":
        return cls(
            feature_mean=np.asarray(values["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(values["feature_std"], dtype=np.float32),
            life_min=float(values["life_min"]),
            life_max=float(values["life_max"]),
        )


class Case1Dataset(Dataset):
    def __init__(self, samples: Sequence[BatterySample], normalizer: Normalizer):
        self.samples = list(samples)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        time_series = torch.from_numpy(self.normalizer.normalize_time(sample.time_series))
        heatmap = torch.from_numpy(sample.heatmap)
        life = torch.tensor(self.normalizer.normalize_life(sample.life_cycles), dtype=torch.float32)
        return time_series, heatmap, life, sample.sample_id


def _parse_name(path: Path) -> Tuple[str, float]:
    if "_life_" not in path.stem:
        raise ValueError(f"Missing '_life_' marker in {path.name}")
    source_name, life_text = path.stem.rsplit("_life_", 1)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", life_text)
    if match is None:
        raise ValueError(f"Cannot parse Life value from {path.name}")
    return source_name, float(match.group(1))


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    old_position = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    new_position = np.linspace(0.0, 1.0, length, dtype=np.float64)
    return np.interp(new_position, old_position, values).astype(np.float32)


def _read_time_series(path: Path, sequence_length: int) -> np.ndarray:
    frame = pd.read_excel(path, sheet_name="cycle_1", engine="openpyxl")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    charge = frame.loc[frame["工步类型"] == CHARGE_STEP, ["电压(V)", "电流(mA)"]].copy()
    charge = charge.apply(pd.to_numeric, errors="coerce").dropna()
    if len(charge) < 10:
        raise ValueError(f"{path.name} contains only {len(charge)} valid charging rows")
    voltage = _resample(charge["电压(V)"].to_numpy(dtype=np.float64), sequence_length)
    current = _resample(charge["电流(mA)"].to_numpy(dtype=np.float64), sequence_length)
    result = np.stack((voltage, current), axis=-1).astype(np.float32)
    if result.shape != (sequence_length, 2) or not np.isfinite(result).all():
        raise ValueError(f"Invalid resampled time series in {path.name}: {result.shape}")
    return result


def _read_heatmap(path: Path, heatmap_size: Tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing heatmap: {path}")
    with Image.open(path) as image:
        image = image.convert("RGB")
        # PIL 的尺寸参数顺序是（宽，高），配置中保存的是（高，宽）。
        image = image.resize((heatmap_size[1], heatmap_size[0]), Image.Resampling.LANCZOS)
        result = np.asarray(image, dtype=np.float32) / 255.0
    result = np.transpose(result, (2, 0, 1)).copy()
    expected = (3, heatmap_size[0], heatmap_size[1])
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError(f"Invalid heatmap {path}: expected {expected}, found {result.shape}")
    return result


def discover_case1_files(data_root: Path) -> List[Path]:
    files = [
        path
        for path in data_root.glob("*.xlsx")
        if path.stem.lower() != "cycle" and "total" not in path.stem.lower()
    ]
    return sorted(files, key=lambda item: item.name)


def load_case1_samples(
    data_root: Path,
    sequence_length: int = 100,
    heatmap_size: Tuple[int, int] = (224, 224),
    expected_count: Optional[int] = 64,
) -> List[BatterySample]:
    """读取全部电池；缺少或格式错误的提交数据会立即报错。"""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Case 1 data directory does not exist: {data_root}")
    files = discover_case1_files(data_root)
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} Case 1 workbooks, found {len(files)}")

    samples: List[BatterySample] = []
    for index, workbook in enumerate(files, start=1):
        source_name, life_cycles = _parse_name(workbook)
        heatmap_path = data_root / source_name / "charge_1" / "cycle_1.png"
        samples.append(
            BatterySample(
                sample_id=workbook.stem,
                source_name=source_name,
                life_cycles=life_cycles,
                time_series=_read_time_series(workbook, sequence_length),
                heatmap=_read_heatmap(heatmap_path, heatmap_size),
            )
        )
        print(f"Loaded Case 1 cell {index:02d}/{len(files)}: {workbook.stem}")
    return samples


def split_case1_samples(
    samples: Sequence[BatterySample], seed: int = 42, train_ratio: float = 0.70
) -> Tuple[List[BatterySample], List[BatterySample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    train_count = int(round(len(ordered) * train_ratio))
    train_indices = set(int(value) for value in permutation[:train_count])
    train = [sample for index, sample in enumerate(ordered) if index in train_indices]
    test = [sample for index, sample in enumerate(ordered) if index not in train_indices]
    return train, test


def apply_saved_split(
    samples: Sequence[BatterySample], train_ids: Iterable[str], test_ids: Iterable[str]
) -> Tuple[List[BatterySample], List[BatterySample]]:
    by_id = {sample.sample_id: sample for sample in samples}
    train_ids = list(train_ids)
    test_ids = list(test_ids)
    expected = set(train_ids) | set(test_ids)
    if expected != set(by_id) or set(train_ids) & set(test_ids):
        raise ValueError("Checkpoint split does not exactly match the discovered Case 1 samples")
    return [by_id[value] for value in train_ids], [by_id[value] for value in test_ids]
