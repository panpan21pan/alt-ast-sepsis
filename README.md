# Analysis code — ALT/AST ratio in sepsis

Companion code for the manuscript:

> **The ALT/AST ratio in sepsis varies with circulatory failure rather than adiposity: implications for fatty-liver indices in critical care**

The code reproduces the cohort construction, descriptive statistics, variance decomposition, Cox models and figures reported in the manuscript.

## Data access

The eICU Collaborative Research Database (eICU-CRD v2.0) and MIMIC-IV (v2.2) are available to credentialed users through PhysioNet (https://physionet.org). The data use agreements prohibit redistribution, so raw data are **not** included here.

Place the source files under:

```
raw_data/资料06- 原始数据及数据处理/
├── D01_baseline_EICU_ICU_first_0626.RData
├── D01_baseline_MIMIC_ICU_first_0626.RData
├── 脓毒症-EICU预后数据(1).csv
└── 脓毒症-mimic预后数据(1).csv
```

## Environment

Python 3.11 with pandas 2.0, NumPy, SciPy 1.11, lifelines 0.30, matplotlib, pyreadr and scikit-learn.

```
pip install -r requirements.txt
```

## Run order

```
python analysis_r1.py        # cohort build, Table 1, Cox models, variance, calibration
mkdir -p analysis_r2 figures
python analysis_r8_final.py  # consolidated estimates -> analysis_r2/primary_results.json
python analysis_r9_consistency.py  # modified vs four-term HSI -> analysis_r2/hsi_consistency.json
python analysis_r6_structure.py  # console checks: correlations, R2, Cox, phenotype
python make_figures.py # Figures 2-4
python make_figure1_flow.py # Figure 1 (flow diagram)
```

`analysis_r1.py` must run first; the remaining scripts read `analysis_r1/cohort_EICU.csv` and `analysis_r1/cohort_MIMIC.csv`.

## Expected key outputs

| Estimate | eICU | MIMIC-IV |
|---|---|---|
| Spearman rho, log(ALT/AST) vs lactate | −0.179 (n=6,431) | −0.163 (n=9,106) |
| R², lactate alone | 0.031 | 0.022 |
| R², adiposity markers | 0.001 | 0.003 |
| ALT/AST HR per SD, base | 0.833 | 0.944 |
| ALT/AST HR per SD, + lactate | 0.837 | 0.960 |
| BMI HR per SD, base | 0.893 | 0.965 |

Full numeric output: `expected_outputs/primary_results.json` and `expected_outputs/hsi_consistency.json`.

## Notes

- All data are de-identified; no protected health information is handled.
- Diabetes ascertainment: 2.1% in eICU (structured admission flags) and 30.2% in MIMIC-IV (ICD codes). The primary index is a prespecified modified HSI that omits the diabetes term; the four-term index is used as a MIMIC-IV sensitivity analysis.
- `analysis_r8_final.py` and `analysis_r9_consistency.py` compute the modified HSI explicitly as `8 * ratio + BMI + 2 * female`. `analysis_r1.py` also stores the four-term HSI (with diabetes) for the sensitivity analysis; do not confuse the two when comparing outputs.
- The two patients excluded for a non-positive recorded length of stay are excluded only from time-to-event analyses (final eICU n = 8,952).

## License

MIT. See `LICENSE`.
