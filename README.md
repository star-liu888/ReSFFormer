# AIO-FE + ReSFFormer
## Ultra-Early and Transferable First-Cycle Battery Life Prediction

> 📬 📦 💻 🔐 **More comprehensive datasets, complete code, and technical
> support are available upon reasonable request.**
>
> ✉️ **Contact:** xinliu1224@hrbeu.edu.cn

This repository accompanies *Ultra-Early and Transferable First-Cycle Battery
Life Prediction for Cell Grouping with AIO-FE and ReSFFormer*.

The released materials provide a compact research implementation for battery
life prediction from first-cycle measurements. The paper and supplementary
material remain the primary references for equations, experimental design, and
architectural interpretation.

---

## 🔬 Background & Motivation

Accurate early battery life estimation is important for electric-vehicle safety,
battery health diagnostics, and predictive maintenance. Conventional workflows
often separate feature construction and life prediction, which can introduce
error propagation and inconsistent degradation representation.

The proposed framework learns informative patterns directly from early-cycle
signals and supports robust prediction under limited-sample conditions.

---

## 🧠 Proposed Method: Targeted Representation Learning

The framework combines temporal measurements with heatmap-derived degradation
information in an end-to-end learning pipeline.

### 1️⃣ Shared Temporal Representation Learning

Early-cycle voltage and current signals are encoded into a shared temporal
representation that captures degradation-related dynamics across charging
profiles.

### 2️⃣ Targeted Feature Fusion

Complementary temporal and heatmap information are fused to improve robustness
when only a small number of labeled cells are available.

### 3️⃣ Stabilized Few-Sample Prediction

ReSFFormer introduces representation-stabilizing attention mechanisms to reduce
noise sensitivity and strengthen generalization across battery datasets.

---

## 🗺️ ReSFFormer

ReSFFormer is designed for few-sample battery life prediction. It combines
condition-selective feature fusion, phase-aware positional alignment, and
dual-domain temporal attention. These components are summarized here at a high
level; detailed derivations and ablation settings are provided in the paper.

Model evaluation follows the reported protocol using RMSE, MAE, and MAPE.

---

## 🔋 Dataset

The experiments cover three battery datasets with different cell sources,
cycling conditions, and lifetime distributions. Together, they are used to
evaluate early-cycle prediction accuracy, representation stability, and
cross-dataset generalization.

Public dataset references are listed in `data/public_datasets.md`.

---

## 💻 Environment

- Python 3.10+
- PyTorch 2.x with a compatible torchvision build
- NumPy, pandas, Pillow, and matplotlib

Install dependencies from `requirements.txt`, then configure the local dataset
path according to your environment.

---

## 📌 Citation

If you use this implementation, please cite the accompanying paper.
