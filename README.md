# Intersectional Fairness in Hiring Algorithms 

Fairness-Aware Resume Ranking Using Deep Learning and Post-processing Calibration

## Project Overview
This project investigates fairness in resume ranking systems using the FairCV dataset. We implement a CNN-based ranker trained with a differential fairness-aware loss, apply the GerryFairClassifier for fairness-constrained classification, and explore post-processing fairness improvements using Multicalibration (HKRR/HJZ).

The goal is to mitigate algorithmic bias across intersectional subgroups (e.g., gender × ethnicity) while maintaining model performance.

---
## 🎛️ Key Components

- 📊**CNN Ranker**: Uses Conv1D + GlobalAveragePooling for ranking tabular features.
- **Differential Fairness**: Applied as a custom loss during training over group IDs.
-📖 **GerryFairClassifier**: Enforces fairness via FP/TP-rate constraints.
-🎚️ **Multicalibration**: Post-processing method using HKRR and HJZ to improve calibration across subgroups.


---


## 🗂️ Dataset
[FairCV dataset](https://arxiv.org/abs/2112.01477) 
FairCV database is a large synthetic dataset without bias created for the express purpose of conducting fairness in Algorithmic hiring research 


---
## 🧰 Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- AIFairness360
- Fairlearn
- multicalibration
- scikit-learn

---
### ✅ 1. Clone the Repository
```bash
git clone https://github.com/your-username/resume-ranking.git
cd resume-ranking
```

### ✅ 2. Create and Activate a Virtual Environment
```bash
python3.9 -m venv .venv
source .venv/bin/activate  # For Unix/macOS
# .venv\Scripts\activate   # For Windows
```

### ✅ 3. Install Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Additional dependencies (if not in `requirements.txt`):
```bash
pip install aif360 tensorflow-ranking scikit-learn matplotlib pandas
```

---

## 🔄 Usage

### 🧪 Run All Stages

```bash
python main.py
```

This runs the full pipeline:
- Preprocessing of FairCV data
- Training CNN ranker with differential fairness loss
- Evaluation (NDCG, ECE, fairness metrics)
- Post-processing with HKRR or HJZ multicalibration

---
## 📈 Evaluation metrics
- Groupwise confusion matrices
- Calibration plots per group
- ε-DF and γ-SF measurements
- Fairness vs. accuracy trade-off curves

---

## 📌 Notes
- Compatible with Python 3.9
- Uses AIF360, TensorFlow Ranking, and multicalibration libraries
- Ensure your terminal or IDE uses the correct virtual environment

---

# DOCUMENTATION

## 📠 Contact
Ananya Krishnakumar - ananya.krishnakumar@mail.mcgill.ca / ananya.krishnakv@gmail.com
Sydney Dacks - hannah.dacks@mail.mcgill.ca / sydney.dacks@gmail.com 

## 💐 Citations
- [FairCV dataset](https://github.com/BiDAlab/FairCVtest)
- [Differential Fairness](https://www.mdpi.com/1099-4300/25/4/660)
- [GerryFair](https://github.com/Trusted-AI/AIF360/tree/main/aif360/algorithms/inprocessing/gerryfair)
- [Multicalibration (HKRR/HJZ)](https://github.com/sid-devic/multicalibration)







