# 🚀 AIO-FE + ReSFFormer
## Ultra-Early First-Cycle Battery Life Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)]()
[![Task](https://img.shields.io/badge/Task-Battery%20Life%20Prediction-0b7285.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Code-success.svg)]()

This repository contains the Case 1 implementation for the paper
*Ultra-Early and Transferable First-Cycle Battery Life Prediction for Cell
Grouping with AIO-FE and ReSFFormer*.

## 🧠 Proposed Framework

The method combines complementary first-cycle signals in a single end-to-end
predictor:

- **AIO-FE** learns multi-scale temporal representations from the first-cycle
  voltage and current traces.
- **Heatmap encoding** extracts spatial degradation cues from the charging
  correlation map with deformable convolutions.
- **CSAF** adaptively modulates the temporal representation with heatmap
  context, producing the fused battery representation.
- **ReSFFormer** models temporal dependencies with phase-aligned rotary
  attention and the dual-domain temporal (DuET) attention mechanism.

The supplied data files contain the 256-dimensional pooled CSAF features and
the corresponding Life values for all 64 Case 1 cells.

## 📁 Repository Layout

```text
config.py                 # Reproducible experiment settings
data.py                   # Case 1 data loading and split utilities
model.py                  # AIO-FE, CSAF, DuET and ReSFFormer
main.py                   # Training entry point
export_features.py        # CSAF feature export and correlation audit
case1_csaf_features.csv   # Exported 256-D CSAF features
case1_life.csv            # Cell Life values and split labels
requirements.txt          # Python dependencies
```

## ▶️ Usage

```bash
python main.py --device cuda
python export_features.py --checkpoint ..\\case1_run\\best.pt --device cuda
```

The training checkpoint is written outside this submission directory. The
export command reports the Life correlation audit for the Top-12 CSAF
dimensions without modifying the features or labels.

## 📌 Citation

If you use this code, please cite the accompanying paper.
