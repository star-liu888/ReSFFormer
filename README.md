# AIO-FE + ReSFFormer
## Ultra-Early and Transferable First-Cycle Battery Life Prediction

This repository contains the Case 1 research implementation accompanying
*Ultra-Early and Transferable First-Cycle Battery Life Prediction for Cell
Grouping with AIO-FE and ReSFFormer*.

The released files support reproducible training and evaluation. The paper and
supplementary material remain the primary references for equations, experiment
settings, and architectural interpretation.

## Background & Motivation

Accurate battery life estimation from early-cycle measurements is important for
electric-vehicle safety, battery health diagnostics, and predictive maintenance.
Conventional workflows often treat degradation modeling as a staged process,
which can introduce error propagation and inconsistent feature usage.

The proposed framework uses first-cycle observations to learn a shared battery
degradation representation, then adapts this representation to life prediction
under limited-sample conditions.

## Proposed Method: Targeted Representation Learning

The method combines temporal measurements and heatmap-derived information in an
end-to-end learning pipeline. A shared representation captures long-term aging
patterns, while targeted modules refine the representation for robust battery
life regression.

The implementation follows the supplementary-material defaults: time-series
input dimension 2, heatmap channels 3, sequence length 100, model dimension 256,
8 attention heads, 3 ReSFFormer layers, feed-forward dimension 1024, dropout
0.1, batch size 8, 200 training epochs, Adam learning rate 2e-4, weight decay
1e-4, and a 10-epoch warm-up cosine schedule.

## ReSFFormer

ReSFFormer is designed for few-sample battery life prediction. It combines
condition-selective feature fusion, phase-aware positional alignment, and
dual-domain temporal attention. These components are described at a high level
here; the detailed derivations and ablation settings are provided in the paper.

The model is evaluated with RMSE, MAE, and MAPE, matching the reported
evaluation protocol.

## Dataset

Case 1 contains 64 laboratory cells tested under a CC-CV charging protocol. The
public dataset references are listed in `data/public_datasets.md`.

## Environment

- Python 3.10+
- PyTorch 2.x with a compatible torchvision build
- NumPy, pandas, Pillow, and matplotlib

Install dependencies from `requirements.txt`, then set the local data path
before training.

## Citation

If you use this implementation, please cite the accompanying paper.
