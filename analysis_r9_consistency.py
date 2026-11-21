"""R9: internal consistency check for modified HSI + MIMIC-only full-HSI sensitivity."""
import warnings, json
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

out = {}
for tag in ["EICU", "MIMIC"]:
    d = pd.read_csv(f"analysis_r1/cohort_{tag}.csv", low_memory=False)
    d["y"] = d.death_inhosp.astype(int)
    d["t"] = d.hosplosday if "hosplosday" in d else d.hosp_day
    d = d[d.t > 0].copy()
    d["hsi_full"] = d.hsi                      # with +2*dm
    d["hsi_nd"] = 8 * d.ratio + d.BMI + 2 * d.female
    d["lac_sd"] = (d.Lactate - d.Lactate.mean()) / d.Lactate.std()
    print(f"\n=== {tag} dm prevalence {d.dm.mean():.3f}")
    print(f"    HSI full mean {d.hsi_full.mean():.2f} | mHSI mean {d.hsi_nd.mean():.2f} | "
          f"patients differing {int((d.hsi_full != d.hsi_nd).sum())}")

    rows = []
    for label, col in [("mHSI", "hsi_nd"), ("full HSI", "hsi_full")]:
        d["x_sd"] = (d[col] - d[col].mean()) / d[col].std()
        for model, extra in [("base", []), ("+lactate", ["lac_sd"])]:
            cols = ["x_sd", "age", "sev_std", "cirr"] + extra + ["t", "y"]
            df = d[cols].dropna()
            c = CoxPHFitter(penalizer=1e-4).fit(df, "t", "y")
            s = c.summary.loc["x_sd"]
            rows.append({"index": label, "model": model, "n": len(df),
                         "HR": round(float(s["exp(coef)"]), 3),
                         "lo": round(float(s["exp(coef) lower 95%"]), 3),
                         "hi": round(float(s["exp(coef) upper 95%"]), 3),
                         "p": float(s["p"])})
    out[tag] = rows
    for r in rows:
        print("   ", r)

json.dump(out, open("analysis_r2/hsi_consistency.json", "w"), indent=1)
