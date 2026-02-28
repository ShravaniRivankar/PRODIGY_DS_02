"""
============================================================
  PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP | TASK 01
  Bar Chart & Histogram: Distribution of Ages & Genders
  Tools: Python, Pandas, Matplotlib, Seaborn
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─────────────────────────────────────────────
# 1. CREATE / LOAD THE DATASET
# ─────────────────────────────────────────────
# If you have the actual CSV from GitHub, replace this block with:
#   df = pd.read_csv("train.csv")   # or whatever the filename is

np.random.seed(42)
n = 1000

# Simulated population dataset
df = pd.DataFrame({
    "Age": np.concatenate([
        np.random.normal(loc=25, scale=5,  size=300),   # young adults
        np.random.normal(loc=40, scale=8,  size=400),   # middle-aged
        np.random.normal(loc=65, scale=10, size=300),   # seniors
    ]).clip(0, 100).astype(int),
    "Gender": np.random.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.48, 0.04])
})

print("Dataset shape  :", df.shape)
print("Age stats:\n",     df["Age"].describe().round(2))
print("\nGender counts:\n", df["Gender"].value_counts())


# ─────────────────────────────────────────────
# 2. STYLING
# ─────────────────────────────────────────────
sns.set_style("dark")
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
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

COLORS = {
    "Male":   "#4fc3f7",
    "Female": "#f48fb1",
    "Other":  "#a5d6a7",
    "hist":   "#7c4dff",
    "kde":    "#ff6d00",
}


# ─────────────────────────────────────────────
# 3. BUILD THE FIGURE  (2 rows × 2 cols)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0f0f1a")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # Bar chart – Gender distribution
ax2 = fig.add_subplot(gs[0, 1])   # Pie chart  – Gender %
ax3 = fig.add_subplot(gs[1, :])   # Histogram  – Age distribution (full width)


# ── Plot 1 : Gender Bar Chart ────────────────
gender_counts = df["Gender"].value_counts()
bars = ax1.bar(
    gender_counts.index,
    gender_counts.values,
    color=[COLORS[g] for g in gender_counts.index],
    edgecolor="#0f0f1a",
    linewidth=1.5,
    width=0.5,
)
for bar, val in zip(bars, gender_counts.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(val),
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="white",
    )
ax1.set_title("Gender Distribution", fontsize=14, fontweight="bold",
              color="white", pad=12)
ax1.set_xlabel("Gender", fontsize=11)
ax1.set_ylabel("Count",  fontsize=11)
ax1.set_ylim(0, gender_counts.max() * 1.18)
ax1.grid(axis="y", alpha=0.4)
ax1.tick_params(axis="both", labelsize=10)


# ── Plot 2 : Gender Pie Chart ────────────────
wedge_colors = [COLORS[g] for g in gender_counts.index]
wedges, texts, autotexts = ax2.pie(
    gender_counts.values,
    labels=gender_counts.index,
    autopct="%1.1f%%",
    colors=wedge_colors,
    startangle=140,
    wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2},
    pctdistance=0.75,
)
for t in texts:
    t.set_color("white"); t.set_fontsize(11)
for at in autotexts:
    at.set_color("white"); at.set_fontsize(9); at.set_fontweight("bold")
ax2.set_title("Gender Share (%)", fontsize=14, fontweight="bold",
              color="white", pad=12)


# ── Plot 3 : Age Histogram + KDE ─────────────
ax3.hist(
    df["Age"],
    bins=30,
    color=COLORS["hist"],
    edgecolor="#0f0f1a",
    alpha=0.75,
    linewidth=0.8,
    label="Age Count",
)

# Overlay KDE on a twin axis so scale matches nicely
ax3_twin = ax3.twinx()
kde_vals = df["Age"].plot.kde(ax=ax3_twin, color=COLORS["kde"],
                               linewidth=2.5, label="KDE Curve")
ax3_twin.set_ylabel("Density", fontsize=11, color=COLORS["kde"])
ax3_twin.tick_params(axis="y", labelcolor=COLORS["kde"])
ax3_twin.set_facecolor("#1a1a2e")
ax3_twin.grid(False)

# Age group shading
age_groups = [(0, 17, "Children\n(0-17)", 0.08),
              (18, 34, "Young Adults\n(18-34)", 0.08),
              (35, 54, "Middle Age\n(35-54)", 0.08),
              (55, 100, "Seniors\n(55+)", 0.08)]
shade_colors = ["#1a237e", "#1b5e20", "#4a148c", "#b71c1c"]

for (lo, hi, label, alpha), sc in zip(age_groups, shade_colors):
    ax3.axvspan(lo, hi, alpha=alpha, color=sc)
    ax3.text((lo + hi) / 2, ax3.get_ylim()[1] * 0.88,
             label, ha="center", va="top",
             fontsize=8, color="white", alpha=0.7)

ax3.set_title("Age Distribution Across the Population", fontsize=14,
              fontweight="bold", color="white", pad=12)
ax3.set_xlabel("Age (years)", fontsize=11)
ax3.set_ylabel("Number of People", fontsize=11)
ax3.legend(loc="upper left", fontsize=10,
           facecolor="#1a1a2e", edgecolor="#3a3a5c", labelcolor="white")
ax3.grid(axis="y", alpha=0.4)


# ── Super title ──────────────────────────────
fig.suptitle(
    "PRODIGY INFOTECH  |  Task-01  –  Population Distribution Analysis",
    fontsize=16, fontweight="bold", color="white", y=0.98
)

plt.savefig("prodigy_task01_output.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✅ Plot saved as prodigy_task01_output.png")
plt.show()
