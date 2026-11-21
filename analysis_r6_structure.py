"""R6: do liver enzymes track shock or adiposity in sepsis?"""
import warnings
import numpy as np, pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

def load(tag):
    d = pd.read_csv(f"analysis_r1/cohort_{tag}.csv", low_memory=False)
    d["y"] = d.death_inhosp.astype(int)
    d["t"] = d.hosplosday if "hosplosday" in d else d.hosp_day
    d["logratio"] = np.log(d.ratio)
    d["logalt"] = np.log(d.ALT.clip(lower=1))
    d["logast"] = np.log(d.AST.clip(lower=1))
    d["mhsi"] = 8 * d.ratio + d.BMI + 2 * d.female   # modified HSI, no diabetes term
    return d[d.t > 0]

for tag in ["EICU", "MIMIC"]:
    d = load(tag)
    print("\n" + "=" * 20, tag, "N =", len(d))
    # 1. Spearman correlations with log ratio
    print("\n-- Spearman rho with log(ALT/AST)")
    for v in ["Lactate", "sev", "BMI", "age", "BilirubinTotal", "Creatinine"]:
        if v not in d: continue
        x = d[[ "logratio", v]].dropna()
        rho, p = stats.spearmanr(x.logratio, x[v])
        print(f"   {v:16s} rho = {rho:+.3f}  p = {p:.2e}  n = {len(x)}")

    # 2. R2 decomposition (OLS on standardized predictors)
    print("\n-- R2 for log(ratio)")
    shock = [c for c in ["Lactate", "sev"] if c in d]
    adip = [c for c in ["BMI", "age", "dm"] if c in d]
    for name, cols in [("shock", shock), ("adiposity", adip), ("both", shock + adip)]:
        x = d[["logratio"] + cols].dropna()
        X = (x[cols] - x[cols].mean()) / x[cols].std()
        X = np.column_stack([np.ones(len(X)), X.values])
        y = x.logratio.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"   {name:10s} R2 = {r2:.3f}  (n={len(x)}, k={len(cols)})")

    # 3. HR attenuation by lactate
    print("\n-- Cox HR per 1 SD of log(ratio), sequential adjustment")
    d["ratio_sd"] = (d.logratio - d.logratio.mean()) / d.logratio.std()
    d["lac_sd"] = (d.Lactate - d.Lactate.mean()) / d.Lactate.std()
    for name, cov in [("base (age,sex,sev,cirr,dm)", ["age", "female", "sev_std", "cirr", "dm"]),
                      ("base + lactate", ["age", "female", "sev_std", "cirr", "dm", "lac_sd"])]:
        cols = ["ratio_sd"] + cov + ["t", "y"]
        df = d[cols].dropna()
        c = CoxPHFitter(penalizer=1e-4).fit(df, "t", "y")
        s = c.summary.loc["ratio_sd"]
        print(f"   {name:28s} HR {s['exp(coef)']:.3f} ({s['exp(coef) lower 95%']:.3f}-{s['exp(coef) upper 95%']:.3f}) n={len(df)}")

    # 4. AST-predominant vs ALT-predominant
    print("\n-- Phenotype by ratio (<1 = AST-predominant)")
    d["pheno"] = np.where(d.ratio < 1, "AST-predominant (<1)", "ALT-predominant (>=1)")
    g = d.groupby("pheno").agg(n=("y", "size"), mort=("y", "mean"),
                               lactate=("Lactate", "median"), sofatot=("sev", "mean"),
                               bmi=("BMI", "mean"), hsi=("mhsi", "mean"),
                               bili=("BilirubinTotal", "median"))
    print(g.round(2).to_string())
