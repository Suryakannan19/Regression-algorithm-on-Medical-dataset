"""
Medical Insurance Cost Predictor
=================================
Full ML pipeline: EDA → Preprocessing → Model Training → Evaluation → Saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─── 1. Generate / Load Dataset ─────────────────────────────────────────────────
print("=" * 60)
print("  MEDICAL INSURANCE COST PREDICTOR — ML PIPELINE")
print("=" * 60)

print("\n[1/6] Loading dataset...")

# Uses the well-known insurance dataset (kaggle compatible)
# If you have the CSV, replace this with: df = pd.read_csv("insurance.csv")
np.random.seed(42)
n = 1338

ages = np.random.randint(18, 65, n)
sexes = np.random.choice(["male", "female"], n)
bmis = np.round(np.random.normal(30.7, 6.1, n), 1).clip(15, 55)
children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
smokers = np.random.choice(["yes", "no"], n, p=[0.20, 0.80])
regions = np.random.choice(["northeast", "northwest", "southeast", "southwest"], n)

# Realistic cost formula with noise
charges = (
    250 * ages
    + 350 * bmis
    + 500 * children
    + (smokers == "yes") * 23000
    + (regions == "southeast") * 1000
    + np.random.normal(0, 2500, n)
).clip(1121, 65000).round(2)

df = pd.DataFrame({
    "age": ages, "sex": sexes, "bmi": bmis,
    "children": children, "smoker": smokers,
    "region": regions, "charges": charges
})

print(f"   Dataset shape: {df.shape}")
print(f"   Features: {list(df.columns[:-1])}")
print(f"   Target: charges (medical insurance cost in USD)")

# ─── 2. EDA ─────────────────────────────────────────────────────────────────────
print("\n[2/6] Exploratory Data Analysis...")

print(f"\n   Charges — Mean: ${df.charges.mean():,.0f} | "
      f"Median: ${df.charges.median():,.0f} | "
      f"Std: ${df.charges.std():,.0f}")
print(f"   Smokers: {(df.smoker=='yes').sum()} ({(df.smoker=='yes').mean()*100:.1f}%)")
print(f"   Avg cost — Smoker: ${df[df.smoker=='yes'].charges.mean():,.0f} | "
      f"Non-smoker: ${df[df.smoker=='no'].charges.mean():,.0f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Medical Insurance Cost — EDA", fontsize=16, fontweight="bold", y=1.01)

# Charges distribution
axes[0, 0].hist(df.charges, bins=40, color="#7b2fff", edgecolor="white", alpha=0.85)
axes[0, 0].set_title("Distribution of Insurance Charges")
axes[0, 0].set_xlabel("Charges (USD)")
axes[0, 0].set_ylabel("Frequency")

# Age vs Charges
scatter = axes[0, 1].scatter(df.age, df.charges,
                              c=(df.smoker == "yes").astype(int),
                              cmap="coolwarm", alpha=0.5, s=15)
axes[0, 1].set_title("Age vs Charges (red=smoker)")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Charges (USD)")

# BMI vs Charges
axes[0, 2].scatter(df.bmi, df.charges,
                   c=(df.smoker == "yes").astype(int),
                   cmap="coolwarm", alpha=0.5, s=15)
axes[0, 2].set_title("BMI vs Charges (red=smoker)")
axes[0, 2].set_xlabel("BMI")
axes[0, 2].set_ylabel("Charges (USD)")

# Smoker vs charges boxplot
df.boxplot(column="charges", by="smoker", ax=axes[1, 0],
           boxprops=dict(color="#7b2fff"), medianprops=dict(color="red"))
axes[1, 0].set_title("Charges by Smoking Status")
axes[1, 0].set_xlabel("Smoker")
plt.sca(axes[1, 0])
plt.title("Charges by Smoking Status")

# Region breakdown
region_means = df.groupby("region")["charges"].mean().sort_values()
axes[1, 1].barh(region_means.index, region_means.values,
                color=["#00d4ff", "#7b2fff", "#ff6b6b", "#ffd700"])
axes[1, 1].set_title("Average Charges by Region")
axes[1, 1].set_xlabel("Average Charges (USD)")

# Correlation heatmap
df_enc = df.copy()
df_enc["sex"] = (df_enc["sex"] == "male").astype(int)
df_enc["smoker"] = (df_enc["smoker"] == "yes").astype(int)
df_enc = pd.get_dummies(df_enc, columns=["region"], drop_first=True)
corr = df_enc.corr()
sns.heatmap(corr[["charges"]].sort_values("charges", ascending=False),
            annot=True, fmt=".2f", cmap="RdYlGn",
            ax=axes[1, 2], cbar=True)
axes[1, 2].set_title("Feature Correlation with Charges")

plt.tight_layout()
plt.savefig("outputs/eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: outputs/eda_plots.png")

# ─── 3. Preprocessing ────────────────────────────────────────────────────────────
print("\n[3/6] Preprocessing...")

df_model = df.copy()
le = LabelEncoder()
df_model["sex"] = le.fit_transform(df_model["sex"])
df_model["smoker"] = le.fit_transform(df_model["smoker"])
df_model = pd.get_dummies(df_model, columns=["region"], drop_first=True)

X = df_model.drop("charges", axis=1)
y = df_model["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"   Features after encoding: {list(X.columns)}")

# ─── 4. Train Multiple Models ────────────────────────────────────────────────────
print("\n[4/6] Training models...")

models = {
    "Linear Regression":     LinearRegression(),
    "Ridge Regression":      Ridge(alpha=10),
    "Lasso Regression":      Lasso(alpha=10),
    "Decision Tree":         DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":         RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR":                   Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=1000, epsilon=500))]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv, "model": model}
    print(f"   {name:<25} R²={r2:.4f}  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  CV_R²={cv:.4f}")

# ─── 5. Best Model + Tuning ──────────────────────────────────────────────────────
print("\n[5/6] Tuning best model (Gradient Boosting)...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}
gb = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_r2   = r2_score(y_test, y_pred_best)
best_mae  = mean_absolute_error(y_test, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"   Best params: {grid_search.best_params_}")
print(f"   Tuned R²={best_r2:.4f}  MAE=${best_mae:,.0f}  RMSE=${best_rmse:,.0f}")

# Save best model
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("   Saved: models/best_model.pkl")

# ─── 6. Results & Plots ──────────────────────────────────────────────────────────
print("\n[6/6] Generating result plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Evaluation — Gradient Boosting (Tuned)", fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_best, alpha=0.4, color="#7b2fff", s=15)
lim = [min(y_test.min(), y_pred_best.min()) - 500,
       max(y_test.max(), y_pred_best.max()) + 500]
axes[0].plot(lim, lim, "r--", lw=2, label="Perfect prediction")
axes[0].set_xlabel("Actual Charges")
axes[0].set_ylabel("Predicted Charges")
axes[0].set_title(f"Actual vs Predicted\nR² = {best_r2:.4f}")
axes[0].legend()

# Residuals
residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=40, color="#00d4ff", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")

# Feature Importances
feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
axes[2].barh(feat_imp.index, feat_imp.values,
             color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_imp))))
axes[2].set_title("Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("outputs/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: outputs/model_evaluation.png")

# Model comparison bar chart
res_df = pd.DataFrame({k: v for k, v in results.items() if k != "model"}).T
res_df_sorted = res_df.sort_values("R2", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(res_df_sorted.index, res_df_sorted["R2"].astype(float),
               color=plt.cm.plasma(np.linspace(0.2, 0.85, len(res_df_sorted))))
ax.set_xlabel("R² Score")
ax.set_title("Model Comparison — R² Score (Test Set)")
ax.axvline(0.8, color="red", linestyle="--", alpha=0.5, label="R²=0.80 baseline")
for bar, val in zip(bars, res_df_sorted["R2"].astype(float)):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: outputs/model_comparison.png")

# ─── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  Best Model  : Gradient Boosting (Tuned)")
print(f"  R² Score    : {best_r2:.4f}  ({best_r2*100:.1f}% variance explained)")
print(f"  MAE         : ${best_mae:,.0f}")
print(f"  RMSE        : ${best_rmse:,.0f}")
print(f"\n  Top feature : {feat_imp.idxmax()} (importance={feat_imp.max():.3f})")
print("\n  Outputs saved in: outputs/ and models/")
print("=" * 60)
