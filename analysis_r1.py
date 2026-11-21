"""R1: cohort rebuild, Table 1, component HRs, variance decomposition, calibration, PH."""
import os, warnings, json
import numpy as np, pandas as pd, pyreadr
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

BASE = "raw_data/资料06- 原始数据及数据处理"
OUT = "analysis_r1"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261118)

def load(tag):
    b = pyreadr.read_r(os.path.join(BASE, f"D01_baseline_{tag}_ICU_first_0626.RData"))["baseline"]
    if tag == "EICU":
        o = pd.read_csv(os.path.join(BASE, "脓毒症-EICU预后数据(1).csv"))
        d = o.merge(b, left_on="patientunitstayid", right_on="ID", how="inner")
        d["sev"] = d.SOFA
        d["death28"] = ((d.hospdischargestatus == "Expired") & (d.hosplosday <= 28)).astype(int)
        d["death_inhosp"] = (d.hospdischargestatus == "Expired").astype(int)
        d["time28"] = np.where(d.death28 == 1, d.hosplosday, np.minimum(d.hosplosday, 28.0))
        d["disch"] = ((d.hospdischargestatus == "Alive") & (d.hosplosday < 28)).astype(int)
    else:
        o = pd.read_csv(os.path.join(BASE, "脓毒症-mimic预后数据(1).csv"))
        d = o.merge(b, left_on="subject_id", right_on="ID", how="inner")
        d["sev"] = d.APSIII
        d["death28"] = ((d.is_hosp_dead == 1) & (d.hosp_day <= 28)).astype(int)
        d["death_inhosp"] = (d.is_hosp_dead == 1).astype(int)
        d["time28"] = np.where(d.death28 == 1, d.hosp_day, np.minimum(d.hosp_day, 28.0))
        d["disch"] = ((d.is_hosp_dead == 0) & (d.hosp_day < 28)).astype(int)
    d = d[(d.Age >= 18) & d.ALT.notna() & d.AST.notna() & (d.AST > 0)
          & d.BMI.notna() & d.BMI.between(15, 60) & d.Gender.notna() & d.sev.notna()].copy()
    d = d[(d.ALT / d.AST).between(0.1, 5.0)].copy()
    d["ratio"] = d.ALT / d.AST
    d["female"] = (d.Gender.astype(str).str.strip() == "Female").astype(int)
    d["dm"] = (d.Diabetes.astype(str).str.strip() == "Yes").astype(int)
    d["cirr"] = (d.Liver_cirrhosis.astype(str).str.strip() == "Yes").astype(int) if "Liver_cirrhosis" in d else 0
    d["vent"] = (d.Ventilation.astype(str).str.strip() == "Yes").astype(int) if "Ventilation" in d else 0
    d["hsi"] = 8 * d.ratio + d.BMI + 2 * d.female + 2 * d.dm
    d["hsi_nobmi"] = 8 * d.ratio + 2 * d.female + 2 * d.dm
    d["age"] = d.Age.astype(float)
    d["sev_std"] = (d.sev - d.sev.mean()) / d.sev.std()
    d["ratio_sd"] = d.ratio / d.ratio.std()
    d["bmi_sd"] = d.BMI / d.BMI.std()
    d["hsi_sd"] = d.hsi / d.hsi.std()
    return d

def table1(d, tag):
    rows = []
    def add(name, a, b=None, fmt="{:.1f} ({:.1f})"):
        rows.append({"Characteristic": name, tag: fmt.format(a, b) if b is not None else f"{a}"})
    add("N", len(d)); add("Age, mean (SD)", d.age.mean(), d.age.std())
    add("Female, %", 100 * d.female.mean())
    add("BMI, mean (SD)", d.BMI.mean(), d.BMI.std())
    add("ALT/AST, median (IQR)", d.ratio.median())
    rows[-1][tag] = f"{d.ratio.median():.2f} ({d.ratio.quantile(.25):.2f}-{d.ratio.quantile(.75):.2f})"
    add("ALT, median (IQR)", d.ALT.median())
    rows[-1][tag] = f"{d.ALT.median():.0f} ({d.ALT.quantile(.25):.0f}-{d.ALT.quantile(.75):.0f})"
    add("AST, median (IQR)", d.AST.median())
    rows[-1][tag] = f"{d.AST.median():.0f} ({d.AST.quantile(.25):.0f}-{d.AST.quantile(.75):.0f})"
    for lab, col in [("Bilirubin, median (IQR)", "BilirubinTotal"), ("Creatinine, median (IQR)", "Creatinine"),
                     ("Lactate, median (IQR)", "Lactate"), ("Platelet, median (IQR)", "PlateletCount"),
                     ("INR, median (IQR)", "INR")]:
        if col in d:
            s = pd.to_numeric(d[col], errors="coerce").dropna()
            rows.append({"Characteristic": f"{lab} [missing {100*(1-len(s)/len(d)):.0f}%]",
                         tag: f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})" if len(s) else "NA"})
    add("Diabetes, %", 100 * d.dm.mean())
    add("Cirrhosis, %", 100 * d.cirr.mean())
    add("Mechanical ventilation, %", 100 * d.vent.mean())
    rows.append({"Characteristic": "Severity score, mean (SD)", tag: f"{d.sev.mean():.1f} ({d.sev.std():.1f})"})
    add("28-day mortality, %", 100 * d.death28.mean())
    add("In-hospital mortality, %", 100 * d.death_inhosp.mean())
    return pd.DataFrame(rows)

def hr_table(d, tag):
    """Cox models: crude and adjusted."""
    cov = ["age", "female", "sev_std", "cirr"]
    specs = {
        "ALT/AST per unit": ("ratio", ["age", "female", "sev_std", "cirr", "dm"]),
        "ALT/AST per SD": ("ratio_sd", ["age", "female", "sev_std", "cirr", "dm"]),
        "BMI per unit": ("BMI", ["age", "female", "sev_std", "cirr", "dm"]),
        "BMI per SD": ("bmi_sd", ["age", "female", "sev_std", "cirr", "dm"]),
        "Four-term HSI per unit": ("hsi", ["age", "sev_std", "cirr"]),      # sex/DM inside HSI
        "Four-term HSI per SD": ("hsi_sd", ["age", "sev_std", "cirr"]),
        "HSI_noBMI per SD": ("hsi_nobmi", ["age", "sev_std"]),
    }
    out = []
    for name, (x, adj) in specs.items():
        for kind, cols in [("crude", [x]), ("adjusted", [x] + [c for c in adj if c != x])]:
            df = d[[c for c in set(cols + ["time28", "death28"])]].dropna()
            keep = [c for c in cols if df[c].nunique() > 1]
            df = df[keep + ["time28", "death28"]]
            cph = CoxPHFitter(penalizer=1e-4).fit(df, "time28", "death28")
            s = cph.summary.loc[x]
            out.append({"db": tag, "model": name, "type": kind, "n": len(df),
                        "HR": s["exp(coef)"], "lo": s["exp(coef) lower 95%"],
                        "hi": s["exp(coef) upper 95%"], "p": s["p"]})
    return pd.DataFrame(out)

def variance_decomp(d, tag):
    w = {"ratio": 8.0, "BMI": 1.0, "female": 2.0, "dm": 2.0}
    cols = list(w)
    X = d[cols].astype(float)
    S = X.cov()
    var_total = sum(w[i] * w[j] * S.loc[i, j] for i in cols for j in cols)
    contrib = {}
    for i in cols:
        own = w[i] ** 2 * S.loc[i, i]
        cov = sum(w[i] * w[j] * S.loc[i, j] for j in cols if j != i)  # full covariance attribution
        contrib[i] = (own, own + cov)
    rows = [{"db": tag, "component": i, "var_own": contrib[i][0],
             "share_own": contrib[i][0] / var_total,
             "share_total": contrib[i][1] / var_total} for i in cols]
    rows.append({"db": tag, "component": "TOTAL", "var_own": var_total, "share_own": 1.0, "share_total": 1.0})
    return pd.DataFrame(rows)

def cindex_ci(d, cols, tag, nboot=200):
    df = d[cols + ["time28", "death28"]].dropna()
    cph = CoxPHFitter().fit(df, "time28", "death28")
    c0 = cph.concordance_index_
    boots = []
    idx = np.arange(len(df))
    for _ in range(nboot):
        s = RNG.choice(idx, len(idx), replace=True)
        if df.iloc[s].death28.sum() < 10: continue
        try:
            boots.append(CoxPHFitter().fit(df.iloc[s], "time28", "death28").concordance_index_)
        except Exception:
            pass
    return {"db": tag, "model": "+".join(cols), "n": len(df), "c": c0,
            "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5)}

def calibration(d, cols, tag):
    df = d[cols + ["time28", "death28"]].dropna().copy()
    cph = CoxPHFitter().fit(df, "time28", "death28")
    lp = cph.predict_partial_hazard(df)
    bh_df = cph.baseline_cumulative_hazard_
    bh = float(np.interp(28.0, bh_df.index.values, bh_df.iloc[:, 0].values))
    risk = 1 - np.exp(-bh * lp.values)
    df["pred"] = np.clip(risk, 1e-6, 1 - 1e-6)
    # Cox calibration slope: outcome on logit(pred)
    df["logit"] = np.log(df.pred / (1 - df.pred))
    cs = CoxPHFitter().fit(df[["logit", "time28", "death28"]], "time28", "death28").params_["logit"]
    # logistic calibration (binary at 28d)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(df[["logit"]], df.death28)
    slope, intercept = lr.coef_[0][0], lr.intercept_[0]
    df["dec"] = pd.qcut(df.pred, 10, labels=False, duplicates="drop")
    dec = df.groupby("dec").agg(n=("death28", "size"), pred=("pred", "mean"), obs=("death28", "mean")).reset_index()
    eo = df.death28.mean() / df.pred.mean()
    return {"db": tag, "model": "+".join(cols), "n": len(df),
            "cox_slope": cs, "logit_slope": slope, "logit_intercept": intercept,
            "mean_pred": df.pred.mean(), "mean_obs": df.death28.mean(), "E_over_O": df.pred.mean() / df.death28.mean()}, dec

def ph_test(d, cols, tag):
    df = d[cols + ["time28", "death28"]].dropna()
    cph = CoxPHFitter().fit(df, "time28", "death28")
    r = proportional_hazard_test(cph, df, time_transform="rank")
    return r.summary.reset_index().assign(db=tag, model="+".join(cols))

if __name__ == "__main__":
    res = {}
    for tag in ["EICU", "MIMIC"]:
        d = load(tag)
        print(f"\n=== {tag}: N={len(d)} 28d death={d.death28.sum()} ({d.death28.mean():.1%}) "
              f"inhosp={d.death_inhosp.sum()} ({d.death_inhosp.mean():.1%})")
        print(f"    SD ratio={d.ratio.std():.3f} BMI={d.BMI.std():.3f} HSI={d.hsi.std():.3f}")
        d.to_csv(f"{OUT}/cohort_{tag}.csv", index=False)
        table1(d, tag).to_csv(f"{OUT}/table1_{tag}.csv", index=False)
        hr_table(d, tag).to_csv(f"{OUT}/hrs_{tag}.csv", index=False)
        variance_decomp(d, tag).to_csv(f"{OUT}/variance_{tag}.csv", index=False)
        rows = []
        cpath = f"{OUT}/cindex_{tag}.csv"
        if os.path.exists(cpath):
            print(f"    [skip cindex, exists] {cpath}")
        else:
            for cols in [["hsi"], ["BMI"], ["ratio"], ["ratio", "BMI"], ["hsi_nobmi", "BMI"], ["hsi", "BMI"]]:
                rows.append(cindex_ci(d, cols, tag, nboot=100))
            pd.DataFrame(rows).to_csv(cpath, index=False)
        cal, dec = calibration(d, ["hsi"], tag)
        json.dump(cal, open(f"{OUT}/calibration_{tag}.json", "w"), indent=1)
        dec.to_csv(f"{OUT}/caldeciles_{tag}.csv", index=False)
        ph_test(d, ["hsi", "BMI", "ratio"], tag).to_csv(f"{OUT}/ph_{tag}.csv", index=False)
    print("\ndone")
