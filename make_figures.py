"""Figures for the ALT/AST manuscript."""
import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300})
R = json.load(open("analysis_r2/primary_results.json"))
C = {"EICU": "#4a6fa5", "MIMIC": "#c0504d"}
L = {"EICU": "eICU", "MIMIC": "MIMIC-IV"}   # display labels

# Figure 1 is produced separately by make_figure1_flow.py.

# ---------- Fig 2: structure ----------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
ax = axes[0]
labels = ["Lactate", "Severity score", "Bilirubin", "Creatinine", "BMI", "Age"]
keys = ["Lactate", "sev", "BilirubinTotal", "Creatinine", "BMI", "age"]
y = np.arange(len(keys))[::-1]
for tag, off in [("EICU", 0.18), ("MIMIC", -0.18)]:
    vals = [R[tag]["corr"][k]["rho"] for k in keys]
    ax.barh(y + off, vals, height=0.34, color=C[tag], label=L[tag])
ax.axvline(0, color="#555", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Spearman rho with log(ALT/AST)"); ax.set_xlim(-0.28, 0.12)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("A   Correlation structure", loc="left", fontweight="bold")

ax = axes[1]
betas = ["Lactate", "sev", "BMI", "age", "female", "dm"]
names = ["Lactate", "Severity", "BMI", "Age", "Female", "Diabetes"]
x = np.arange(len(betas)); w = 0.38
for i, tag in enumerate(["EICU", "MIMIC"]):
    vals = [R[tag]["beta"][b] for b in betas]
    ax.bar(x + (i - 0.5) * w, vals, w, color=C[tag], label=L[tag])
ax.axhline(0, color="#555", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7.5)
ax.set_ylabel("Standardized beta"); ax.legend(frameon=False, fontsize=7.5)
ax.set_title("B   Multivariable structure", loc="left", fontweight="bold")

ax = axes[2]
comps = ["Lactate", "Severity", "Lactate+\nseverity", "Adiposity", "BMI alone"]
keys2 = ["lactate", "severity", "severity_lactate", "adiposity", "bmi"]
x = np.arange(len(comps))
for i, tag in enumerate(["EICU", "MIMIC"]):
    vals = [100 * R[tag]["r2b"][k] for k in keys2]
    ax.bar(x + (i - 0.5) * 0.34, vals, 0.34, color=C[tag], label=L[tag])
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 0.5) * 0.34, v + 0.12, f"{v + 1e-9:.1f}", ha="center", fontsize=6.2)
ax.set_xticks(x); ax.set_xticklabels(comps, fontsize=6.4, rotation=20, ha="right")
ax.set_ylabel("R² for log(ALT/AST) (%)"); ax.legend(frameon=False, fontsize=7.5)
ax.set_ylim(0, 7.5)
ax.set_title("C   Variance explained", loc="left", fontweight="bold")
plt.tight_layout(); fig.savefig("figures/Figure2_structure.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 3: gradient ----------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for ax, key, ylab, title in [(axes[0], "mort", "In-hospital mortality (%)", "A   Mortality"),
                             (axes[1], "lac", "Median lactate (mmol/L)", "B   Lactate"),
                             (axes[2], "bmi", "Mean BMI (kg/m²)", "C   BMI")]:
    for tag in ["EICU", "MIMIC"]:
        q = R[tag]["quintiles"]
        xs = [d["q"] + 1 for d in q]
        ys = [d[key] * (100 if key == "mort" else 1) for d in q]
        ax.plot(xs, ys, "o-", color=C[tag], lw=1.6, ms=5, label=L[tag])
    ax.set_xlabel("Quintile of ALT/AST ratio"); ax.set_ylabel(ylab)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)
axes[2].set_ylim(27.5, 30.5)
plt.tight_layout(); fig.savefig("figures/Figure3_gradient.png", bbox_inches="tight"); plt.close(fig)

# ---------- Fig 4: phenotype ----------
metrics = [("mort", "Mortality (%)", 100), ("lac", "Lactate (mmol/L)", 1),
           ("sev", "Severity score", 1), ("bmi", "BMI (kg/m²)", 1), ("hsi", "HSI", 1)]
fig, axes = plt.subplots(2, len(metrics), figsize=(12, 5.2))
for r, tag in enumerate(["EICU", "MIMIC"]):
    ph = {d["pheno"]: d for d in R[tag]["phenotype"]}
    for c, (key, ylab, mult) in enumerate(metrics):
        ax = axes[r, c]
        vals = [ph["ALTpred"][key] * mult, ph["ASTpred"][key] * mult]
        ax.bar([0, 1], vals, color=["#7fa7c9", "#c0504d"], width=0.62)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ALT>\nAST", "AST>\nALT"], fontsize=7)
        ax.set_title(f"{L[tag]}: {ylab}", fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v + 1e-9:.1f}", ha="center", va="bottom", fontsize=7)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(0, hi * 1.22)
        ax.axhline(0, color="#999", lw=0.6)
plt.tight_layout(); fig.savefig("figures/Figure4_phenotype.png", bbox_inches="tight"); plt.close(fig)
print("done")
