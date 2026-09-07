"""导出 64 节电池的 pooled CSAF 特征和 Life，并审计 Top-12。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import Case1Config, make_parser, set_seed
from data import Case1Dataset, Normalizer, load_case1_samples, split_case1_samples
from model import build_model


def _load_checkpoint(path: Path, config: Case1Config, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _contribution_scores(model, dataset: Case1Dataset, device: torch.device) -> np.ndarray:
    """计算全部 64 个样本的平均绝对梯度乘激活贡献度。"""
    total = np.zeros(256, dtype=np.float64)
    for index in range(len(dataset)):
        time_series, heatmap, _, _ = dataset[index]
        time_series = time_series.unsqueeze(0).to(device)
        heatmap = heatmap.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        prediction, intermediate = model(time_series, heatmap, return_intermediates=True)
        f_fuse = intermediate["f_fuse"]
        gradient = torch.autograd.grad(prediction.sum(), f_fuse, retain_graph=False)[0]
        total += (gradient * f_fuse).abs().detach().cpu().numpy()[0]
    return total / max(1, len(dataset))


def export_features(config: Case1Config, checkpoint_path: Path, device_name: str | None = None) -> Dict[str, object]:
    set_seed(config.seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples = load_case1_samples(
        config.data_root,
        sequence_length=config.sequence_length,
        heatmap_size=config.heatmap_size,
        expected_count=64,
    )
    train_samples, test_samples = split_case1_samples(samples, config.seed, config.train_ratio)
    normalizer = Normalizer.fit(train_samples)
    dataset = Case1Dataset(samples, normalizer)
    model = _load_checkpoint(checkpoint_path, config, device)

    contributions = _contribution_scores(model, dataset, device)
    top12 = np.argsort(-contributions, kind="stable")[:12]
    features: List[np.ndarray] = []
    metadata: List[Tuple[str, str, float]] = []
    train_ids = {sample.sample_id for sample in train_samples}
    with torch.no_grad():
        for sample in samples:
            time_series = torch.from_numpy(normalizer.normalize_time(sample.time_series)).unsqueeze(0).to(device)
            heatmap = torch.from_numpy(sample.heatmap).unsqueeze(0).to(device)
            _, intermediate = model(time_series, heatmap, return_intermediates=True)
            features.append(intermediate["f_fuse"].cpu().numpy()[0])
            metadata.append((sample.sample_id, "train" if sample.sample_id in train_ids else "test", sample.life_cycles))

    feature_array = np.asarray(features, dtype=np.float32)
    feature_columns = [f"csaf_{index:03d}" for index in range(feature_array.shape[1])]
    feature_frame = pd.DataFrame(feature_array, columns=feature_columns)
    feature_frame.insert(0, "split", [item[1] for item in metadata])
    feature_frame.insert(0, "sample_id", [item[0] for item in metadata])

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "case1_csaf_features.csv"
    life_path = output_dir / "case1_life.csv"
    feature_frame.to_csv(feature_path, index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "sample_id": [item[0] for item in metadata],
            "life_cycles": [item[2] for item in metadata],
            "split": [item[1] for item in metadata],
        }
    ).to_csv(life_path, index=False, encoding="utf-8")

    lives = np.asarray([item[2] for item in metadata], dtype=np.float64)
    correlations = np.asarray(
        [np.corrcoef(feature_array[:, index], lives)[0, 1] for index in range(feature_array.shape[1])]
    )
    passing = np.abs(correlations[top12]) >= 0.80
    result = {
        "feature_file": str(feature_path),
        "life_file": str(life_path),
        "n_samples": int(len(samples)),
        "n_features": int(feature_array.shape[1]),
        "top12_indices": [int(index) for index in top12],
        "top12_correlations": [float(correlations[index]) for index in top12],
        "top12_abs_r_ge_0_80": int(passing.sum()),
        "threshold": 4,
        "status": "PASS" if int(passing.sum()) >= 4 else "NOT_MET",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    parser = make_parser("Export Case 1 CSAF features and Life values")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = Case1Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_dir=args.run_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    export_features(config, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
