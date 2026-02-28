"""
============================================================
  PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP | TASK 02
  Data Cleaning & Exploratory Data Analysis (EDA)
  Dataset: Titanic (Kaggle)
  Tools: Python, Pandas, Matplotlib, Seaborn
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CREATE TITANIC-LIKE DATASET
# ─────────────────────────────────────────────
# If you have the real CSV, replace with:
#   df = pd.read_csv("train.csv")

np.random.seed(42)
n = 891  # same size as real Titanic training set

pclass  = np.random.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
sex     = np.random.choice(["male", "female"], size=n, p=[0.65, 0.35])
age     = np.where(
    np.random.rand(n) < 0.2, np.nan,
    np.clip(np.random.normal(29, 14, n), 1, 80)
)
fare    = np.where(
    pclass == 1,
    np.random.exponential(80, n),
    np.where(pclass == 2, np.random.exponential(20, n),
             np.random.exponential(10, n))
)
embarked = np.random.choice(["S", "C", "Q", None], size=n, p=[0.72, 0.19, 0.08, 0.01])

# Survival: higher for female, 1st class
survive_prob = (
    0.15
    + 0.35 * (sex == "female")
    + 0.20 * (pclass == 1)
    - 0.10 * (pclass == 3)
    + np.random.normal(0, 0.05, n)
).clip(0, 1)
survived = (np.random.rand(n) < survive_prob).astype(int)

df = pd.DataFrame({
    "PassengerId": range(1, n + 1),
    "Survived":   survived,
    "Pclass":     pclass,
    "Sex":        sex,
    "Age":        age.round(1),
    "Fare":       fare.round(2),
    "Embarked":   embarked,
    "SibSp":      np.random.choice([0,1,2,3], size=n, p=[0.68,0.23,0.07,0.02]),
    "Parch":      np.random.choice([0,1,2,3], size=n, p=[0.76,0.13,0.08,0.03]),
})

print("=" * 55)
print("  PRODIGY INFOTECH | TASK 02 — EDA on Titanic Dataset")
print("=" * 55)


# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n📋 STEP 1: Initial Data Overview")
print(f"   Shape       : {df.shape}")
print(f"   Columns     : {list(df.columns)}")
print("\n   Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
df["AgeGroup"]   = pd.cut(df["Age"],
                           bins=[0, 12, 18, 35, 60, 100],
                           labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

print("\n✅ STEP 2: After Cleaning")
print(f"   Missing Values Remaining: {df.isnull().sum().sum()}")
print(f"\n   Survival Rate: {df['Survived'].mean()*100:.1f}%")
print(f"   Male   survival: {df[df['Sex']=='male']['Survived'].mean()*100:.1f}%")
print(f"   Female survival: {df[df['Sex']=='female']['Survived'].mean()*100:.1f}%")


# ─────────────────────────────────────────────
# 3. STYLING
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#3a3a5c",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
})

SURVIVED_COLORS = ["#ef5350", "#66bb6a"]   # red = died, green = survived
CLASS_COLORS    = ["#ffd54f", "#4fc3f7", "#ce93d8"]
SEX_COLORS      = ["#4fc3f7", "#f48fb1"]


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS  (3×2 grid)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f0f1a")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])  # Survival count
ax2 = fig.add_subplot(gs[0, 1])  # Survival by Gender
ax3 = fig.add_subplot(gs[1, 0])  # Survival by Pclass
ax4 = fig.add_subplot(gs[1, 1])  # Age distribution
ax5 = fig.add_subplot(gs[2, 0])  # Fare distribution by class
ax6 = fig.add_subplot(gs[2, 1])  # Correlation heatmap


# ── Plot 1: Overall Survival Count ───────────
surv_counts = df["Survived"].value_counts().sort_index()
bars = ax1.bar(["Did Not Survive", "Survived"],
               surv_counts.values,
               color=SURVIVED_COLORS,
               edgecolor="#0f0f1a", linewidth=1.5, width=0.5)
for bar, val in zip(bars, surv_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             f"{val}\n({val/n*100:.1f}%)",
             ha="center", fontsize=11, fontweight="bold", color="white")
ax1.set_title("Overall Survival Count", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Number of Passengers")
ax1.set_ylim(0, surv_counts.max() * 1.2)
ax1.grid(axis="y")


# ── Plot 2: Survival Rate by Gender ──────────
gender_surv = df.groupby("Sex")["Survived"].mean() * 100
bars = ax2.bar(gender_surv.index, gender_surv.values,
               color=SEX_COLORS, edgecolor="#0f0f1a",
               linewidth=1.5, width=0.4)
for bar, val in zip(bars, gender_surv.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{val:.1f}%",
             ha="center", fontsize=12, fontweight="bold", color="white")
ax2.set_title("Survival Rate by Gender", fontsize=13, fontweight="bold", pad=10)
ax2.set_ylabel("Survival Rate (%)")
ax2.set_ylim(0, 100)
ax2.grid(axis="y")


# ── Plot 3: Survival by Passenger Class ──────
class_surv = df.groupby("Pclass")["Survived"].mean() * 100
bars = ax3.bar([f"Class {c}" for c in class_surv.index],
               class_surv.values,
               color=CLASS_COLORS, edgecolor="#0f0f1a",
               linewidth=1.5, width=0.5)
for bar, val in zip(bars, class_surv.values):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{val:.1f}%",
             ha="center", fontsize=12, fontweight="bold", color="white")
ax3.set_title("Survival Rate by Passenger Class", fontsize=13, fontweight="bold", pad=10)
ax3.set_ylabel("Survival Rate (%)")
ax3.set_ylim(0, 100)
ax3.grid(axis="y")


# ── Plot 4: Age Distribution by Survival ─────
for surv, color, label in zip([0, 1], SURVIVED_COLORS,
                                ["Did Not Survive", "Survived"]):
    ax4.hist(df[df["Survived"] == surv]["Age"],
             bins=25, alpha=0.65, color=color,
             edgecolor="#0f0f1a", linewidth=0.5, label=label)
ax4.set_title("Age Distribution by Survival", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Age")
ax4.set_ylabel("Count")
ax4.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="white")
ax4.grid(axis="y")


# ── Plot 5: Fare Distribution by Class ───────
for i, (cls, color) in enumerate(zip([1, 2, 3], CLASS_COLORS)):
    data = df[df["Pclass"] == cls]["Fare"]
    ax5.hist(data, bins=20, alpha=0.7, color=color,
             edgecolor="#0f0f1a", linewidth=0.5,
             label=f"Class {cls}")
ax5.set_title("Fare Distribution by Class", fontsize=13, fontweight="bold", pad=10)
ax5.set_xlabel("Fare (£)")
ax5.set_ylabel("Count")
ax5.set_xlim(0, 300)
ax5.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="white")
ax5.grid(axis="y")


# ── Plot 6: Correlation Heatmap ───────────────
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare",
             "FamilySize", "IsAlone"]
corr = df[corr_cols].corr()

im = ax6.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax6.set_xticks(range(len(corr_cols)))
ax6.set_yticks(range(len(corr_cols)))
ax6.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
ax6.set_yticklabels(corr_cols, fontsize=8)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr.iloc[i, j]
        ax6.text(j, i, f"{val:.2f}",
                 ha="center", va="center",
                 fontsize=7,
                 color="white" if abs(val) > 0.3 else "#aaaaaa")
plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
ax6.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=10)


# ── Super title ───────────────────────────────
fig.suptitle(
    "PRODIGY INFOTECH  |  Task-02  —  Titanic Dataset: Data Cleaning & EDA",
    fontsize=15, fontweight="bold", color="white", y=0.98
)

plt.savefig("prodigy_task02_output.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✅ Plot saved as prodigy_task02_output.png")
plt.show()
