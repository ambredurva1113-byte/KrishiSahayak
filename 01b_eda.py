"""
KrishiSahayak — Exploratory Data Analysis
==========================================

Satisfies assignment Section 5B "Data Understanding":
    - Dataset description       -> printed block below
    - Variable description      -> printed block below
    - Exploratory Data Analysis -> plots saved to eda_outputs/
    - Summary statistics        -> eda_outputs/summary_statistics.csv
    - Data visualization        -> eda_outputs/*.png

Also covers the "Outlier detection" item from Section 5C (Data
Pre-processing), which is a separate rubric line from EDA but uses the
same IQR machinery, so it lives here rather than in 01_generate_data.py.

Run this AFTER 01_generate_data.py (it needs krishisahayak_dataset.csv)
and BEFORE 02_train_model.py.

    python 01_generate_data.py
    python 01b_eda.py
    python 02_train_model.py

Everything printed to the console in this script is written so it can
be copied more or less directly into your report's "Dataset Description",
"Exploratory Data Analysis", and "Data Pre-processing / Outlier Detection"
sections.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "eda_outputs"
OUT_DIR.mkdir(exist_ok=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load RAW data (before cleaning) — this is what "Data Understanding"
#    should describe, since it's the state of the data before you touched it
# ---------------------------------------------------------------------------
section("1. RAW DATASET — BEFORE CLEANING")

raw = pd.read_csv(BASE_DIR / "raw_final_crop.csv")
raw = raw.rename(columns={
    "Dist Name": "district", "Year": "year", "Crop": "crop",
    "Area(1000 ha)": "area_1000ha", "Production(1000 tons)": "production_1000t",
    "Yield(Kg per ha)": "yield_kg_per_ha", "Total Rainfall": "rainfall_mm",
    "Avg Temp": "avg_temp_c",
})
raw = raw.drop(columns=[c for c in raw.columns if "Unnamed" in c])

print(f"Shape                 : {raw.shape[0]} rows x {raw.shape[1]} columns")
print(f"Districts             : {raw['district'].nunique()}")
print(f"Crops                 : {raw['crop'].nunique()}")
print(f"Year range            : {raw['year'].min()} - {raw['year'].max()}")
print(f"Duplicate district-year-crop rows: "
      f"{raw.duplicated(subset=['district', 'year', 'crop']).sum()}")

# ---- Variable description table (paste into report) -----------------------
var_desc = pd.DataFrame({
    "Variable": ["district", "year", "crop", "area_1000ha", "production_1000t",
                 "yield_kg_per_ha", "rainfall_mm", "avg_temp_c"],
    "Type": ["Categorical", "Numeric (discrete)", "Categorical", "Numeric (continuous)",
             "Numeric (continuous)", "Numeric (continuous)", "Numeric (continuous)",
             "Numeric (continuous)"],
    "Description": [
        "Maharashtra district name",
        "Agricultural year of record",
        "Crop grown (e.g. Cotton, Rice, Sugarcane)",
        "Area sown, in thousand hectares",
        "Total production, in thousand tonnes",
        "Yield = production / area, in kg per hectare (this is the raw basis for the target)",
        "Total seasonal rainfall in mm for that district-year",
        "Average temperature in Celsius for that district-year",
    ],
})
print("\nVariable description:")
print(var_desc.to_string(index=False))
var_desc.to_csv(OUT_DIR / "variable_description.csv", index=False)

# ---- Missing values (on the RAW file, before any dropna) -------------------
section("2. MISSING VALUE ANALYSIS (raw file, before cleaning)")
missing = raw.isna().sum()
missing_pct = (missing / len(raw) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0]
print(missing_report if len(missing_report) else "No missing values found.")
missing_report.to_csv(OUT_DIR / "missing_value_report.csv")

plt.figure(figsize=(6, 4))
sns.heatmap(raw.isna(), cbar=False, cmap="Reds", yticklabels=False)
plt.title("Missing Value Map — Raw Dataset")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_missing_value_map.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------------------
section("3. SUMMARY STATISTICS (numeric columns, raw file)")
numeric_cols_raw = ["area_1000ha", "production_1000t", "yield_kg_per_ha",
                     "rainfall_mm", "avg_temp_c"]
summary = raw[numeric_cols_raw].describe().T
summary["skew"] = raw[numeric_cols_raw].skew()
print(summary.round(2))
summary.round(2).to_csv(OUT_DIR / "summary_statistics.csv")

# ---------------------------------------------------------------------------
# 4. Outlier detection (IQR method) — Section 5C requirement
# ---------------------------------------------------------------------------
section("4. OUTLIER DETECTION (IQR method, raw file, non-null rows)")


def iqr_outliers(series):
    s = series.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lower) | (s > upper)).sum()
    return n_out, len(s), lower, upper


outlier_rows = []
for col in numeric_cols_raw:
    n_out, n_total, lower, upper = iqr_outliers(raw[col])
    outlier_rows.append({
        "variable": col, "n_outliers": n_out, "pct_outliers": round(n_out / n_total * 100, 2),
        "lower_bound": round(lower, 2), "upper_bound": round(upper, 2),
    })
outlier_report = pd.DataFrame(outlier_rows)
print(outlier_report.to_string(index=False))
outlier_report.to_csv(OUT_DIR / "outlier_report.csv", index=False)
print(
    "\nDecision: outliers are NOT removed. In this domain a very high or very "
    "low yield/rainfall value is often a genuine drought or bumper-crop year "
    "— exactly the signal this project is trying to detect — so deleting "
    "them would remove the distress events we care about. This is stated "
    "explicitly rather than silently keeping or dropping them."
)

plt.figure(figsize=(10, 4))
for i, col in enumerate(["rainfall_mm", "yield_kg_per_ha", "area_1000ha"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=raw[col], color="#66BB6A")
    plt.title(col)
plt.suptitle("Outlier Check — Boxplots (IQR method)")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_outlier_boxplots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Univariate distributions
# ---------------------------------------------------------------------------
section("5. UNIVARIATE DISTRIBUTIONS")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(raw["rainfall_mm"].dropna(), bins=30, kde=True, ax=axes[0], color="#1E88E5")
axes[0].set_title("Rainfall Distribution (mm)")
sns.histplot(raw["avg_temp_c"].dropna(), bins=30, kde=True, ax=axes[1], color="#FB8C00")
axes[1].set_title("Avg Temperature Distribution (°C)")
sns.histplot(raw["yield_kg_per_ha"].dropna(), bins=30, kde=True, ax=axes[2], color="#43A047")
axes[2].set_title("Yield Distribution (kg/ha)")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_univariate_distributions.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Trends over time (state-level aggregate)
# ---------------------------------------------------------------------------
section("6. YEAR-WISE TRENDS (Maharashtra state average)")

yearly = raw.groupby("year").agg(
    avg_yield=("yield_kg_per_ha", "mean"),
    avg_rainfall=("rainfall_mm", "mean"),
).reset_index()
print(yearly.round(1).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(yearly["year"], yearly["avg_yield"], marker="o", color="#2E7D32")
axes[0].set_title("Average Yield Over Time (kg/ha)")
axes[0].set_xlabel("Year")
axes[1].plot(yearly["year"], yearly["avg_rainfall"], marker="o", color="#1565C0")
axes[1].set_title("Average Rainfall Over Time (mm)")
axes[1].set_xlabel("Year")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_year_trends.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Crop and district comparisons
# ---------------------------------------------------------------------------
section("7. CROP AND DISTRICT COMPARISONS")

top_crops = raw["crop"].value_counts().head(8).index
plt.figure(figsize=(9, 5))
sns.boxplot(data=raw[raw["crop"].isin(top_crops)], x="yield_kg_per_ha", y="crop",
            hue="crop", palette="YlGn", legend=False)
plt.title("Yield Distribution by Crop (8 most-recorded crops)")
plt.xlabel("Yield (kg/ha)")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_yield_by_crop.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. Final (model-ready) dataset — target variable and feature correlations
# ---------------------------------------------------------------------------
section("8. FINAL MODEL-READY DATASET (krishisahayak_dataset.csv)")

final = pd.read_csv(BASE_DIR / "krishisahayak_dataset.csv")
print(f"Shape           : {final.shape[0]} rows x {final.shape[1]} columns")
print(f"Districts       : {final['district'].nunique()}")
print(f"Crops           : {final['crop'].nunique()}")
print(f"Year range      : {final['year'].min()} - {final['year'].max()}")
print("\nTarget variable (risk_level) distribution:")
print(final["risk_level"].value_counts())
print("\n(Note: classes are near-balanced by construction — risk_level is built "
      "from tertiles of distress_score — so accuracy is a fair headline metric "
      "here and macro-F1 in 02_train_model.py additionally guards against any "
      "residual imbalance.)")

plt.figure(figsize=(5, 4))
final["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).plot(
    kind="bar", color=["#43A047", "#FB8C00", "#E53935"])
plt.title("Target Variable Distribution — Risk Level")
plt.ylabel("Number of district-crop-year records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "06_target_distribution.png", dpi=150)
plt.close()

# ---- Correlation heatmap of predictor features + target proxy -------------
corr_cols = ["area_1000ha", "rainfall_mm", "avg_temp_c", "rainfall_deviation_pct",
             "prev_year_yield", "prior_3yr_avg_yield", "prior_3yr_yield_volatility",
             "distress_score"]
corr = final[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Heatmap — Predictors and Distress Score")
plt.tight_layout()
plt.savefig(OUT_DIR / "07_correlation_heatmap.png", dpi=150)
plt.close()

strongest = corr["distress_score"].drop("distress_score").abs().sort_values(ascending=False)
print("\nFeatures most correlated with distress_score (magnitude):")
print(strongest.round(3))

# ---- Highest-risk districts (average distress score) ----------------------
district_risk = final.groupby("district")["distress_score"].mean().sort_values(ascending=False)
print("\nTop 10 highest average-risk districts:")
print(district_risk.head(10).round(3))

plt.figure(figsize=(8, 6))
district_risk.head(10).sort_values().plot(kind="barh", color="#C62828")
plt.title("Top 10 Districts by Average Distress Score")
plt.xlabel("Mean distress score (0-1)")
plt.tight_layout()
plt.savefig(OUT_DIR / "08_top_risk_districts.png", dpi=150)
plt.close()

section("DONE")
print(f"All EDA outputs saved to: {OUT_DIR}")
print("Copy the printed tables above into your report's Dataset Description, "
      "EDA, and Outlier Detection sections. Copy the PNGs into your report's "
      "Appendix / EDA section and presentation slides.")
