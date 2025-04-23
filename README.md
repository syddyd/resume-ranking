# Intersectional Fairness in Hiring Algorithms 

Fairness-Aware Resume Ranking Using Deep Learning and Post-processing Calibration

## Project Overview
This project investigates fairness in resume ranking systems using the FairCV dataset. We implement a CNN-based ranker trained with a differential fairness-aware loss, apply the GerryFairClassifier for fairness-constrained classification, and explore post-processing fairness improvements using Multicalibration (HKRR/HJZ).

The goal is to mitigate algorithmic bias across intersectional subgroups (e.g., gender × ethnicity) while maintaining model performance.

## 🎛️ Key Components

- 📊**CNN Ranker**: Uses Conv1D + GlobalAveragePooling for ranking tabular features.
- **Differential Fairness**: Applied as a custom loss during training over group IDs.
- 📖**GerryFairClassifier**: Enforces fairness via FP/TP-rate constraints.
- 🎚️**Multicalibration**: Post-processing method using HKRR and HJZ to improve calibration across subgroups.


## TF-Ranking Model


## 🗂️ Dataset


## 🧰 Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- AIFairness360
- Fairlearn
- multicalibration
- scikit-learn

## 🔌 Installation

# Create virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


## 	🔑 Usage


## 📈 Evaluation metrics
- Groupwise confusion matrices
- Calibration plots per group
- ε-DF and γ-SF measurements
- Fairness vs. accuracy trade-off curves

#DOCUMENTATION

## 📠 Contact


## 💐 Citations
- [FairCV dataset](https://arxiv.org/abs/2112.01477)
- [Differential Fairness](https://arxiv.org/abs/2106.09276)
- [GerryFair](https://github.com/algofairness/gerryfair)
- [Multicalibration (HKRR/HJZ)](https://arxiv.org/abs/1807.06209)
