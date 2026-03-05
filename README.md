# 🏥 Medical Insurance Cost Predictor

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


