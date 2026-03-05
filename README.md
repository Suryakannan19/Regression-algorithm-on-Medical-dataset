# 🏥 Medical Insurance Cost Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A complete end-to-end **regression ML pipeline** to predict individual medical insurance charges based on patient demographics and health indicators. Covers the full data science workflow from EDA to model deployment.

---

## 🎯 Problem Statement

Medical insurance costs vary widely across individuals. This project builds a regression model to predict annual insurance charges using features like age, BMI, smoking status, and region — enabling insurers and individuals to estimate costs accurately.

---

## 📊 Dataset

- **1338 records** (based on the [Kaggle Medical Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance))
- **Features:** age, sex, bmi, children, smoker, region
- **Target:** `charges` — annual insurance cost in USD

---

## 🧠 Models Trained

| Model | R² | MAE |
|-------|----|-----|
| Linear Regression | ~0.75 | ~$4,200 |
| Ridge Regression | ~0.75 | ~$4,200 |
| Lasso Regression | ~0.74 | ~$4,300 |
| Decision Tree | ~0.82 | ~$3,100 |
| Random Forest | ~0.87 | ~$2,700 |
| **Gradient Boosting** *(tuned)* | **~0.89** | **~$2,400** |
| SVR | ~0.76 | ~$3,800 |

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/medical-insurance-predictor.git
cd medical-insurance-predictor
pip install -r requirements.txt
```

### Run the full pipeline
```bash
python train.py
```

### Predict for new patients
```bash
python predict.py
```

### Explore the Jupyter notebook
```bash
jupyter notebook notebook.ipynb
```

---

## 🗂️ Project Structure

```
medical-insurance-predictor/
│
├── train.py             # Full ML pipeline (EDA → Train → Evaluate → Save)
├── predict.py           # Inference script for new patients
├── notebook.ipynb       # Step-by-step Jupyter walkthrough
├── requirements.txt     # Dependencies
├── README.md
│
├── models/              # Saved model files (generated after training)
│   ├── best_model.pkl
│   └── feature_names.pkl
│
└── outputs/             # Generated plots (generated after training)
    ├── eda_plots.png
    ├── model_evaluation.png
    └── model_comparison.png
```

---

## 🔍 Key Findings

- **Smoking** is by far the strongest predictor — smokers pay ~3× more on average
- **BMI** and **age** have strong positive correlations with charges
- **Gradient Boosting** outperforms all linear and tree-based baselines after tuning
- The model explains ~89% of variance in insurance charges (R² ≈ 0.89)

---

## 🛠️ Tech Stack

- **scikit-learn** — ML models, preprocessing, cross-validation, GridSearchCV
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — EDA visualizations
- **joblib** — Model serialization

---

## 📄 License

MIT License
