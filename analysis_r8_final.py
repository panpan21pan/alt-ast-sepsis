"""R8: consolidated numbers for the ALT/AST manuscript."""
import warnings, json
import numpy as np, pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
warnings.filterwarnings("ignore")

res = {}
for tag in ["EICU", "MIMIC"]:
    d = pd.read_csv(f"analysis_r1/cohort_{tag}.csv", low_memory=False)
    d["y"] = d.death_inhosp.astype(int)
    d["t"] = d.hosplosday if "hosplosday" in d else d.hosp_day
    d = d[d.t > 0].copy()
    d["logratio"] = np.log(d.ratio)
    d["ratio_sd"] = (d.logratio - d.logratio.mean()) / d.logratio.std()
    d["hsi_nd"] = 8 * d.ratio + d.BMI + 2 * d.female   # prespecified modified HSI (no diabetes term)
    d["hsi_sd"] = (d.hsi_nd - d.hsi_nd.mean()) / d.hsi_nd.std()
    d["bmi_sd"] = (d.BMI - d.BMI.mean()) / d.BMI.std()
    d["lac_sd"] = (d.Lactate - d.Lactate.mean()) / d.Lactate.std()
    R = {}

    # correlations
    corr = {}
    for v in ["Lactate", "sev", "BilirubinTotal", "Creatinine", "BMI", "age"]:
        x = d[["logratio", v]].dropna()
        rho, p = stats.spearmanr(x.logratio, x[v])
        corr[v] = {"rho": round(float(rho), 3), "p": float(p), "n": int(len(x))}
    R["corr"] = corr

    # R2
    r2 = {}
    for name, cols in [("shock", ["Lactate", "sev"]), ("adiposity", ["BMI", "age", "dm"]),
                       ("both", ["Lactate", "sev", "BMI", "age", "dm"])]:
        x = d[["logratio"] + cols].dropna()
        Z = (x[cols] - x[cols].mean()) / x[cols].std()
        X = np.column_stack([np.ones(len(Z)), Z.values])
        beta, *_ = np.linalg.lstsq(X, x.logratio.values, rcond=None)
        pred = X @ beta
        r2[name] = round(float(1 - ((x.logratio - pred) ** 2).sum() / ((x.logratio - x.logratio.mean()) ** 2).sum()), 3)
    R["r2"] = r2

    # detailed R2 for the four prespecified models (used in Figure 2C)
    r2b = {}
    for name, cols in [("lactate", ["Lactate"]), ("severity", ["sev"]),
                       ("severity_lactate", ["Lactate", "sev"]),
                       ("adiposity", ["BMI", "age", "dm"]), ("bmi", ["BMI"])]:
        x = d[["logratio"] + cols].dropna()
        Z = (x[cols] - x[cols].mean()) / x[cols].std()
        X = np.column_stack([np.ones(len(Z)), Z.values])
        beta, *_ = np.linalg.lstsq(X, x.logratio.values, rcond=None)
        pred = X @ beta
        r2b[name] = round(float(1 - ((x.logratio - pred) ** 2).sum() /
                                ((x.logratio - x.logratio.mean()) ** 2).sum()), 5)
    R["r2b"] = r2b

    # standardized betas
    cols = ["Lactate", "sev", "BMI", "age", "female", "dm"]
    x = d[["logratio"] + cols].dropna()
    Z = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones(len(Z)), Z[cols].values])
    beta, *_ = np.linalg.lstsq(X, Z.logratio.values, rcond=None)
    R["beta"] = {c: round(float(b), 3) for c, b in zip(cols, beta[1:])}

    # quintiles
    d["q"] = pd.qcut(d.ratio, 5, labels=False, duplicates="drop")
    g = d.groupby("q").agg(n=("y", "size"), ratio=("ratio", "median"), mort=("y", "mean"),
                           lac=("Lactate", "median"), bmi=("BMI", "mean"), sofa=("sev", "mean"))
    R["quintiles"] = g.round(3).reset_index().to_dict("records")

    # phenotype
    d["pheno"] = np.where(d.ratio < 1, "ASTpred", "ALTpred")
    g = d.groupby("pheno").agg(n=("y", "size"), mort=("y", "mean"), lac=("Lactate", "median"),
                               sev=("sev", "mean"), bmi=("BMI", "mean"), hsi=("hsi_nd", "mean"),
                               alt=("ALT", "median"), ast=("AST", "median"), bili=("BilirubinTotal", "median"),
                               vent=("vent", "mean"))
    R["phenotype"] = g.round(3).reset_index().to_dict("records")

    # obesity
    d["obese"] = (d.BMI >= 30).astype(int)
    g = d.groupby("obese").agg(n=("y", "size"), ratio=("ratio", "median"), hsi=("hsi_nd", "mean"),
                               mort=("y", "mean"), lac=("Lactate", "median"))
    R["obesity"] = g.round(3).reset_index().to_dict("records")

    # Cox
    cox = []
    for var, label, adj in [("ratio_sd", "ALT/AST per SD", ["age", "female", "sev_std", "cirr", "dm"]),
                            ("hsi_sd", "Modified HSI per SD", ["age", "sev_std", "cirr"]),
                            ("bmi_sd", "BMI per SD", ["age", "female", "sev_std", "cirr", "dm"])]:
        for mname, extra in [("base", []), ("+lactate", ["lac_sd"])]:
            cols = [var] + adj + extra + ["t", "y"]
            df = d[cols].dropna()
            c = CoxPHFitter(penalizer=1e-4).fit(df, "t", "y")
            s = c.summary.loc[var]
            cox.append({"var": label, "model": mname, "n": int(len(df)),
                        "HR": round(float(s["exp(coef)"]), 3),
                        "lo": round(float(s["exp(coef) lower 95%"]), 3),
                        "hi": round(float(s["exp(coef) upper 95%"]), 3),
                        "p": float(s["p"])})
    R["cox"] = cox

    res[tag] = R

json.dump(res, open("analysis_r2/primary_results.json", "w"), indent=1)
for tag in res:
    print("=" * 16, tag)
    print(json.dumps(res[tag], indent=1)[:1800])
