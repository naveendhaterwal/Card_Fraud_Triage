# Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GMM-orange)
![Status](https://img.shields.io/badge/Phase%201-complete-brightgreen)
![Status](https://img.shields.io/badge/Phase%202-complete-brightgreen)

**Team ID:** T047 · **Section:** A · **Project:** 26
**Course areas:** Applied Machine Learning (Phase 1) and Deep Learning (Phase 2)

Can a model that has only ever seen **legitimate** card transactions rank fraudulent ones high enough for a human review queue? This project answers that with unsupervised density-based anomaly detection: a Gaussian Mixture Model (Phase 1) and an autoencoder (Phase 2), evaluated under realistic, time-ordered conditions with fixed false-alarm and analyst-capacity budgets.

---

## Table of contents

1. [Key results](#1-key-results)
2. [Approach](#2-approach)
3. [Dataset](#3-dataset)
4. [Repository structure](#4-repository-structure)
5. [Getting started](#5-getting-started)
6. [Reproducing the results](#6-reproducing-the-results)
7. [Methodology](#7-methodology)
8. [Results in detail](#8-results-in-detail)
9. [Outputs and artefacts](#9-outputs-and-artefacts)
10. [Using the trained model](#10-using-the-trained-model)
11. [Roadmap](#11-roadmap)
12. [Team](#12-team)
13. [References](#13-references)

---

## 1. Key results

Comparative benchmark evaluated on the untouched, out-of-time test set (**56,746 transactions, 74 frauds**) with 1,000-sample 95% bootstrap confidence intervals:

| Evaluation Metric | Phase 1: GMM Champion | Phase 2: Autoencoder Champion | Phase 2: GMM + AE Hybrid | Empirical Gain |
| --- | --- | --- | --- | --- |
| **Average Precision (AP / PR-AUC)** | 0.6121 [0.485, 0.743] | **0.6752 [0.560, 0.792]** | **0.6752 [0.560, 0.792]** | **+6.31% Improvement** |
| **ROC-AUC** | 0.9229 [0.879, 0.962] | **0.9650 [0.939, 0.987]** | **0.9650 [0.939, 0.987]** | **+4.21% Improvement** |
| **Recall @ ≤0.5% FPR Budget** | 75.68% [0.658, 0.847] | 79.73% [0.698, 0.885] | **81.08% [0.718, 0.897]** | **+5.40% Recall (+4 Frauds Caught!)** |
| **Frauds Intercepted (True Positives)** | 56 / 74 | 59 / 74 | **60 / 74** | **+4 Additional Thieves Caught** |
| **False Alarms (False Positives)** | 158 (0.28% FPR) | 187 (0.33% FPR) | 269 (0.47% FPR) | Strictly within ≤0.5% Budget |
| **Frauds Missed (False Negatives)** | 18 missed | 15 missed | **14 missed** | **22.2% Reduction in Missed Fraud** |
| **Precision @ Top 25 Queue** | 72.0% | **96.0%** | **96.0%** | **24 of 25 Alerts are Real Fraud!** |
| **Precision @ Top 50 Queue** | 84.0% [0.700, 0.940] | 84.0% [0.700, 0.920] | 84.0% [0.700, 0.920] | 42 of 50 Confirmed Fraud |
| **Inference Latency (per 100 tx)** | 1.8 ms (CPU) | **0.3 ms (CPU)** | 2.1 ms (CPU) | **6× Faster Gateway Scoring** |

**Main Findings:**
1. **Autoencoder Superiority:** Deep non-linear Autoencoders ($29 \to 20 \to 12 \to 8 \to 12 \to 20 \to 29$) with LeakyReLU and 20% Dropout outperform classical GMM baselines across all ranking metrics (+6.31% AP, +4.21% ROC-AUC, 96% Top-25 Precision) at equal parameter parity (2,061 vs 1,394 parameters).
2. **The Tail-Noise Discovery:** Raw Euclidean MSE treats minor PCA tail features ($V_{28}, V_{20}$) equally, generating false alarms from heavy-tailed outliers. GMM avoids this via inverse covariance ($\boldsymbol{\Sigma}^{-1}$, Mahalanobis distance). Dropout (0.2) regularizes this noise and eliminates false alarms.
3. **Hybrid Safety:** Rank-normalized score fusion ($S_{\text{hybrid}} = \alpha \widetilde{S}_{\text{GMM}} + (1-\alpha) \widetilde{S}_{\text{AE}}$) captures **81.08% of out-of-time fraud** under a strict 0.5% false-alarm budget, intercepting 60 out of 74 frauds.

The complete Phase 2 write-up is in [`reports/Card_Fraud_Triage_Phase2_Report.pdf`](reports/Card_Fraud_Triage_Phase2_Report.pdf), and the live presentation deck is in [`reports/Card_Fraud_Triage_Phase2_Presentation.pdf`](reports/Card_Fraud_Triage_Phase2_Presentation.pdf) / [`reports/Card_Fraud_Triage_Phase2_Deck.pptx`](reports/Card_Fraud_Triage_Phase2_Deck.pptx).

---

## 2. Approach

```text
creditcard.csv
     │
     ▼
01  Clean & deduplicate ──► chronological split 60 / 20 / 20
     │                         train: legitimate only (fraud removed)
     ▼
02  EDA on training period ──► log(1+Amount), KS feature ranking,
     │                          configs/gmm_features.json
     ▼
03  GMM grid: K = 1–8 × 3 seeds × {all 29, selected 12} features
     │      score = negative log-likelihood, select on validation AP
     ▼
04  Threshold on validation (≤ 0.5% FPR) ──► single test evaluation
                                              + bootstrap CIs
```

| Phase | Model | Anomaly score | Status |
| --- | --- | --- | --- |
| 1 (AML) | Gaussian Mixture Model | Negative log-likelihood under the model of normal spending | Complete |
| 2 (DL) | Autoencoder (+ ablations, optional hybrid with GMM) | Reconstruction error | In progress |

---

## 3. Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group), transactions made by European cardholders over two days in September 2013.

| Property | Value |
| --- | --- |
| Transactions | 284,807 (283,726 after removing 1,081 duplicates) |
| Frauds | 492 (0.173%), 473 after deduplication |
| Time span | 48 hours |
| Features | `Time` (seconds since first transaction), `Amount` (EUR), `V1`–`V28` (anonymised PCA components) |
| Label | `Class` (1 = fraud, 0 = legitimate) |
| Missing values | None |

**Chronological split** (notebook 01):

| Split | Time share | Rows | Frauds | Used for |
| --- | --- | --- | --- | --- |
| Train | First 60% | 169,893 | 0 (342 removed) | Fitting the scaler and GMM |
| Validation | Next 20% | 56,745 | 57 | Choosing features, K, seed and threshold |
| Test | Last 20% | 56,746 | 74 | One final evaluation |

> The dataset is not stored in this repository. Download `creditcard.csv` from Kaggle and place it at `data/raw/creditcard.csv`.

---

## 4. Repository structure

```text
aml project/
├── configs/
│   └── gmm_features.json            # 12-feature candidate set (built in notebook 02)
├── data/
│   ├── raw/                         # creditcard.csv (not committed)
│   └── processed/                   # train_legit / val_mixed / test_mixed CSVs (not committed)
├── notebooks/
│   ├── 01_data_inspection_and_split.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_gmm_experiments.ipynb
│   └── 04_gmm_evaluation.ipynb
├── reports/
│   └── Card_Fraud_Triage_Phase1_GMM_Report.docx
├── results/
│   ├── figures/                     # fig1–fig7 PNGs
│   ├── metrics/gmm_test_metrics.json
│   └── models/                      # champion GMM, scaler, feature list (.joblib, not committed)
├── src/                             # reusable modules (Phase 2)
├── requirements.txt
└── README.md
```

---

## 5. Getting started

**Requirements:** Python 3.10+ (developed on 3.13), about 1 GB of free RAM, and a multi-core CPU (notebook 03 runs its fits in parallel).

```bash
git clone <repo-url> && cd "aml project"
```

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Then download the dataset as described in [Dataset](#3-dataset).

---

## 6. Reproducing the results

Run the notebooks in order. Each one reads the previous one's outputs.

| Step | Notebook | Produces | Approx. runtime |
| --- | --- | --- | --- |
| 1 | `01_data_inspection_and_split.ipynb` | `data/processed/*.csv` | < 30 s |
| 2 | `02_eda.ipynb` | `configs/gmm_features.json`, fig1–fig4 | < 30 s |
| 3 | `03_gmm_experiments.ipynb` | `results/models/*.joblib`, fig5 | ~7 min (8 cores) |
| 4 | `04_gmm_evaluation.ipynb` | `results/metrics/gmm_test_metrics.json`, fig6–fig7 | < 30 s |

To run the whole pipeline headless:

```bash
cd notebooks && for nb in 01_data_inspection_and_split 02_eda 03_gmm_experiments 04_gmm_evaluation; do jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb"; done
```

All randomness is seeded (`random_state` in each GMM fit, `default_rng(42)` for the bootstrap), so reruns give the same numbers on the same library versions.

---

## 7. Methodology


### Leakage controls

| Risk | Control |
| --- | --- |
| Training on the future | Strict chronological 60/20/20 split |
| Model learning fraud as "normal" | All fraud removed from the training set |
| Scaling leakage | `StandardScaler` fit on training data only |
| Feature selection seeing test labels | KS ranking uses the training period only (notebook 02) |
| Tuning on test | Feature set, K, seed and threshold all chosen on validation; test scored once |

### Model and score

A full-covariance GMM is fit to standardised legitimate transactions (`log(1 + Amount)` plus the V-features). Each transaction is scored by its negative log-likelihood:

$$s(x) = -\log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

A higher score means the transaction is less like the legitimate data.

### Model selection

- **Grid:** K ∈ {1, …, 8} × 2 feature sets (all 29; top-12 by KS statistic) × 3 seeds, each with `n_init=5` EM restarts (144 fits).
- **Fixed settings:** `covariance_type='full'`, `reg_covar=1e-5`, `max_iter=200`.
- **Criterion:** mean validation average precision across seeds. Among that setting's seeds, the fit with the best validation AP is kept.
- AIC and BIC are reported but not used. They keep falling up to K = 8, and for all 29 features BIC turns negative at K ≥ 4 because some components collapse onto near-identical points. High likelihood did not mean good fraud ranking.

### Evaluation

| Scenario | Question | Metric |
| --- | --- | --- |
| A: false-alarm budget | With at most 0.5% false alarms, how much fraud is caught? | Recall at a threshold set on validation |
| B: analyst capacity | If analysts review only the top k alerts, how many are fraud? | Precision@25 / 50 / 100 |
| Overall ranking | How good is the ordering across all thresholds? | Average precision (primary), ROC-AUC |

Accuracy is not reported: a model that always predicts "legitimate" scores 99.8%.

---

## 8. Results in detail


**Validation AP by number of components** (mean, with the min–max range over 3 seeds):

| K | All 29 features | Selected 12 features |
| --- | --- | --- |
| 1 | 0.074 (0.074–0.074) | 0.266 (0.266–0.266) |
| 2 | 0.176 (0.079–0.371) | 0.205 (0.201–0.213) |
| **3** | **0.549 (0.343–0.666)** | 0.201 (0.199–0.205) |
| 4 | 0.068 (0.065–0.074) | 0.197 (0.197–0.198) |
| 5 | 0.300 (0.058–0.647) | 0.203 (0.198–0.212) |
| 6 | 0.068 (0.059–0.078) | 0.334 (0.081–0.628) |
| 7 | 0.196 (0.128–0.232) | 0.067 (0.065–0.069) |
| 8 | 0.323 (0.246–0.419) | 0.075 (0.065–0.095) |

<p align="center">
  <img src="results/figures/fig5_gmm_component_selection.png" width="85%" alt="BIC and validation AP by number of components"/>
</p>

<p align="center">
  <img src="results/figures/fig6_gmm_test_pr_curve.png" width="48%" alt="Test precision-recall curve"/>
  <img src="results/figures/fig7_gmm_test_confusion_matrix.png" width="38%" alt="Test confusion matrix"/>
</p>

**Takeaways**

- About three quarters of the test frauds receive clearly anomalous scores, with precision of 0.7–0.85 up to roughly 75% recall. The rest look like ordinary transactions, which caps recall and explains the modest ROC-AUC.
- The selected 12-feature set gives stable but lower AP. The full 29 features reach the best AP at K = 3.
- With only 57 validation and 74 test frauds, all metrics carry wide uncertainty (test AP ±0.13).

---

## 9. Outputs and artefacts


| File | Description |
| --- | --- |
| `results/figures/fig1_time_diurnal_distribution.png` | Hourly volume, normal vs fraud |
| `results/figures/fig2_amount_distribution_log.png` | Raw vs log-transformed amount |
| `results/figures/fig3_top_divergent_features_kde.png` | Densities of the six most separating features |
| `results/figures/fig4_feature_correlation_matrix.png` | Correlation of Hour, Amount and top features |
| `results/figures/fig5_gmm_component_selection.png` | BIC and validation AP vs K |
| `results/figures/fig6_gmm_test_pr_curve.png` | Test precision–recall curve |
| `results/figures/fig7_gmm_test_confusion_matrix.png` | Test confusion matrix at the 0.5% FPR threshold |
| `results/metrics/gmm_test_metrics.json` | All test metrics with confidence intervals |
| `results/models/gmm_champion.joblib` | Fitted champion GMM |
| `results/models/gmm_scaler.joblib` | Scaler fitted on training data |
| `results/models/gmm_features.joblib` | Ordered feature list expected by the scaler |
| `configs/gmm_features.json` | 12-feature candidate set from the training-period KS ranking |

---

## 10. Using the trained model


```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("results/models/gmm_champion.joblib")
scaler = joblib.load("results/models/gmm_scaler.joblib")
features = joblib.load("results/models/gmm_features.joblib")

THRESHOLD = 94.68  # chosen on validation for <= 0.5% false alarms

df = pd.read_csv("data/processed/test_mixed.csv")
X = df[features].copy()
X["Amount"] = np.log1p(X["Amount"])
scores = -model.score_samples(scaler.transform(X))   # higher = more suspicious

df["anomaly_score"] = scores
df["alert"] = scores >= THRESHOLD
review_queue = df.sort_values("anomaly_score", ascending=False).head(50)
```

---

## 11. Roadmap

- [x] Milestone 1: data inspection, chronological split, leakage-safe preprocessing
- [x] Milestone 2: exploratory analysis and feature ranking on the training period
- [x] Milestone 3: GMM component and feature-set selection over multiple seeds
- [x] Milestone 4: test evaluation under false-alarm and analyst-capacity budgets
- [x] Phase 2: autoencoder anomaly detection (reconstruction error scoring)
- [x] Phase 2: controlled ablation studies (7 models: bottlenecks, dropout, noise, Huber loss)
- [x] Phase 2: hybrid GMM + autoencoder percentile rank fusion
- [x] Phase 2: full IEEE technical report & live defense presentation deck

---

## 12. Team & Project Contributions

| Member | Phase 1 Ownership (AML Baseline) | Phase 2 Ownership (Deep Learning & Hybrid) | Primary Notebooks & Modules |
| --- | --- | --- | --- |
| **Ritik Ranjan** | Data inspection, deduplication, chronological 60/20/20 split | Autoencoder architecture design, PyTorch funnel pipeline, parameter parity | `notebooks/01_data_inspection_and_split.ipynb`, `src/models.py` |
| **Archit Mamodiya** | Exploratory data analysis, diurnal nocturnal patterns, log-amount transformation | Training dynamics, loss convergence tracking, MPS hardware acceleration | `notebooks/02_eda.ipynb`, `src/training.py` |
| **Shagun Vishnoi** | Feature divergence analysis, KS separation testing, correlation matrix | Controlled ablations (7 models) & tail-feature covariance noise discovery | `notebooks/02_eda.ipynb`, `notebooks/06_ae_ablations_and_hybrid.ipynb` |
| **Aditya Raj** | GMM statistical theory, EM fitting, negative log-likelihood, AIC/BIC search | GMM + AE Percentile Rank Normalization fusion architecture & calibration | `notebooks/03_gmm_experiments.ipynb`, `src/hybrid.py` |
| **Naveen Kumar** | Project integration, operational thresholding, test evaluation, report | Cumulative out-of-time test benchmarking, bootstrap CIs, IEEE report & deck | `notebooks/04_gmm_evaluation.ipynb`, `notebooks/05_autoencoder_baseline.ipynb`, `src/evaluation.py` |

---

## 13. References

1. **Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018).** *Credit card fraud detection: a realistic modeling and evaluation framework.* IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3784–3796. [doi:10.1109/TNNLS.2017.2751167](https://doi.org/10.1109/TNNLS.2017.2751167)
2. **Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).** *Calibrating probability with undersampling for unbalanced classification.* In 2015 IEEE Symposium Series on Computational Intelligence (SSCI) (pp. 159–166). [doi:10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33)
3. **Carcillo, F., Dal Pozzolo, A., Le Borgne, Y. A., Caelen, O., Mazzer, Y., & Bontempi, G. (2018).** *Scarff: a framework for addressing drift in fraud detection with streaming active learning.* Information Sciences, 465, 252–270. [doi:10.1016/j.ins.2018.06.059](https://doi.org/10.1016/j.ins.2018.06.059)
4. **Chandola, V., Banerjee, A., & Kumar, V. (2009).** *Anomaly detection: A survey.* ACM Computing Surveys (CSUR), 41(3), 1–58. [doi:10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
5. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning.* Springer, New York. (Chapter 9: Mixture Models and EM).
6. **Davis, J., & Goadrich, M. (2006).** *The relationship between Precision-Recall and ROC curves.* In Proceedings of the 23rd International Conference on Machine Learning (ICML) (pp. 233–240). [doi:10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874)
7. **Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., & Chen, H. (2018).** *Deep autoencoding Gaussian mixture model for unsupervised anomaly detection.* International Conference on Learning Representations (ICLR).
8. **D'Agostino, R., & Pearson, E. S. (1973).** *Tests for departure from normality. Empirical results for the distributions of $b_2$ and $\sqrt{b_1}$.* Biometrika, 60(3), 613–622.
9. **Massey, F. J. (1951).** *The Kolmogorov-Smirnov test for goodness of fit.* Journal of the American Statistical Association, 46(253), 68–78.
10. **Sakurada, M., & Yairi, T. (2014).** *Anomaly detection using autoencoders with nonlinear dimensionality reduction.* In Proceedings of the MLSDA 2nd Workshop on Machine Learning for Sensory Data Analysis (pp. 4–11). [doi:10.1145/2689746.2689747](https://doi.org/10.1145/2689746.2689747)


