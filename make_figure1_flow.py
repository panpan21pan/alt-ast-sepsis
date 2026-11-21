"""Figure 1: study flow, including the time-to-event step."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PANELS = {
    "eICU": {
        "top": "Sepsis ICU admissions\nn = 13,832",
        "steps": [
            "Age < 18 years\nn = 10 excluded",
            "ALT or AST unavailable\nn = 2,785 excluded",
            "BMI missing or outside 15–60 kg/m²\nn = 465 excluded",
            "Sex or severity score missing\nn = 1,574 excluded",
            "ALT/AST ratio outside 0.1–5.0\nn = 44 excluded",
        ],
        "cohort": "Analytic cohort\nn = 8,954",
        "los": "Non-positive length of stay\nn = 2 excluded",
        "final": "Time-to-event cohort\nn = 8,952",
    },
    "MIMIC-IV": {
        "top": "Sepsis ICU admissions\nn = 18,502",
        "steps": [
            "Age < 18 years\nn = 16 excluded",
            "ALT or AST unavailable\nn = 4,328 excluded",
            "BMI missing or outside 15–60 kg/m²\nn = 429 excluded",
            "Sex or severity score missing\nn = 28 excluded",
            "ALT/AST ratio outside 0.1–5.0\nn = 46 excluded",
        ],
        "cohort": "Analytic cohort\nn = 9,655",
        "los": "Non-positive length of stay\nn = 0 excluded",
        "final": "Time-to-event cohort\nn = 9,655",
    },
}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), dpi=300)
for ax, (name, p) in zip(axes, PANELS.items()):
    ax.set_xlim(0, 10)
    ax.set_ylim(5.3, 21.2)
    ax.axis("off")
    ax.set_title(name, fontsize=15, fontweight="bold", pad=10)

    def box(y, text, face="#FFFFFF", edge="#8A8A8A", bold=False, h=1.30):
        ax.add_patch(FancyBboxPatch(
            (1.0, y), 8.0, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            linewidth=1.0, edgecolor=edge, facecolor=face))
        ax.text(5.0, y + h / 2, text, ha="center", va="center",
                fontsize=9.8, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(y_top, y_bottom):
        ax.annotate("", xy=(5.0, y_bottom), xytext=(5.0, y_top),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#8A8A8A",
                                    shrinkA=0, shrinkB=0))

    y = 19.4
    box(y, p["top"], face="#E8EEF7", edge="#4a6fa5", bold=True, h=1.45)
    cursor = y
    for step in p["steps"]:
        arrow(cursor, cursor - 0.55)
        y = cursor - 0.55 - 1.15
        box(y, step)
        cursor = y
    # analytic cohort
    arrow(cursor, cursor - 0.55)
    y = cursor - 0.55 - 1.30
    box(y, p["cohort"], face="#E6F0E6", edge="#4F7F4F", bold=True, h=1.45)
    cursor = y
    # LOS exclusion
    arrow(cursor, cursor - 0.50)
    y = cursor - 0.50 - 1.05
    box(y, p["los"], h=1.20)
    cursor = y
    # final
    arrow(cursor, cursor - 0.50)
    y = cursor - 0.50 - 1.30
    box(y, p["final"], face="#E6F0E6", edge="#4F7F4F", bold=True, h=1.45)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("figures/Figure1_flow.png", bbox_inches="tight", facecolor="white")
print("saved figures/Figure1_flow.png")
