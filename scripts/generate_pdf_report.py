import os
import subprocess
import pypdfium2 as pdfium

def generate_report():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Card Fraud Triage via Gaussian Mixture Density Scoring</title>
<style>
  @page {
    size: letter;
    margin: 16mm 14mm 18mm 14mm;
    @bottom-center {
      content: counter(page);
      font-size: 9pt;
      font-family: 'Times New Roman', Times, serif;
    }
  }

  * {
    box-sizing: border-box;
  }

  body {
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 9.5pt;
    line-height: 1.25;
    color: #111;
    margin: 0;
    padding: 0;
    text-align: justify;
  }

  .header-banner {
    font-size: 7.5pt;
    color: #555;
    border-bottom: 0.5pt solid #888;
    padding-bottom: 2pt;
    margin-bottom: 12pt;
    display: flex;
    justify-content: space-between;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .title {
    text-align: center;
    font-size: 18pt;
    font-weight: bold;
    margin-bottom: 6pt;
    line-height: 1.2;
  }

  .subtitle {
    text-align: center;
    font-size: 11pt;
    font-style: italic;
    color: #444;
    margin-bottom: 10pt;
  }

  .authors {
    text-align: center;
    font-size: 10pt;
    margin-bottom: 4pt;
    font-weight: 600;
  }

  .affiliations {
    text-align: center;
    font-size: 8.5pt;
    color: #444;
    margin-bottom: 14pt;
    font-style: italic;
  }

  .abstract-box {
    margin: 0 18pt 14pt 18pt;
    padding: 8pt 10pt;
    background-color: #fbfbfb;
    border-left: 2.5pt solid #24415c;
    border-right: 0.5pt solid #ddd;
    border-top: 0.5pt solid #ddd;
    border-bottom: 0.5pt solid #ddd;
    font-size: 8.5pt;
    line-height: 1.3;
  }

  .abstract-title {
    font-weight: bold;
    font-style: italic;
    display: inline;
  }

  .keywords {
    margin-top: 5pt;
    font-size: 8pt;
    color: #333;
  }
  .keywords b {
    font-style: italic;
  }

  .columns {
    column-count: 2;
    column-gap: 16pt;
    column-fill: balance;
  }

  h1 {
    font-size: 10.5pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 10pt;
    margin-bottom: 4pt;
    border-bottom: 0.5pt solid #333;
    padding-bottom: 1.5pt;
    break-after: avoid;
  }

  h2 {
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 8pt;
    margin-bottom: 3pt;
    break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 5pt;
    text-indent: 10pt;
  }

  p.no-indent {
    text-indent: 0;
  }

  .dropcap {
    float: left;
    font-size: 28pt;
    line-height: 24pt;
    padding-top: 2pt;
    padding-right: 3pt;
    font-family: 'Times New Roman', Times, serif;
    font-weight: bold;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 8pt 0;
    font-size: 8pt;
    break-inside: avoid;
  }

  table.booktabs {
    border-top: 1.5pt solid #111;
    border-bottom: 1.5pt solid #111;
  }

  table.booktabs th {
    border-bottom: 0.75pt solid #111;
    padding: 3pt 3pt;
    font-weight: bold;
    text-align: left;
  }

  table.booktabs td {
    padding: 2.5pt 3pt;
    border-bottom: 0.25pt solid #eee;
  }

  table.booktabs tr.highlight {
    background-color: #eef4f8;
    font-weight: bold;
  }

  .table-caption {
    font-size: 8pt;
    font-weight: bold;
    margin-bottom: 2pt;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .figure-container {
    width: 100%;
    margin: 7pt 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-container img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ccc;
    border-radius: 2pt;
  }

  .figure-caption {
    font-size: 7.8pt;
    color: #222;
    margin-top: 3pt;
    text-align: justify;
    line-height: 1.2;
  }
  .figure-caption b {
    font-weight: bold;
  }

  .math-block {
    text-align: center;
    margin: 5pt 0;
    font-style: italic;
    font-size: 9pt;
    break-inside: avoid;
  }

  ol, ul {
    margin-top: 2pt;
    margin-bottom: 6pt;
    padding-left: 14pt;
    font-size: 8.8pt;
  }

  li {
    margin-bottom: 2pt;
  }

  .references {
    font-size: 7.8pt;
    line-height: 1.25;
    break-inside: avoid;
  }

  .references ol {
    padding-left: 12pt;
    margin-top: 4pt;
  }

  .references li {
    margin-bottom: 3.5pt;
  }

  .badge {
    display: inline-block;
    padding: 1pt 3pt;
    font-size: 7pt;
    font-weight: bold;
    border-radius: 2pt;
    background: #e2eaf0;
    color: #1a364a;
  }
</style>
</head>
<body>

<div class="header-banner">
  <span>Applied Machine Learning · Project 26 Technical Report</span>
  <span>Team T047 · Section A · October 2026</span>
</div>

<div class="title">Card Fraud Triage via Gaussian Mixture Density Scoring and Unsupervised Anomaly Detection</div>
<div class="subtitle">Phase 1 Research Monograph: Methodological Foundations, Validation Protocol, and Out-of-Time Test Benchmarking</div>

<div class="authors">
  Ritik Ranjan &nbsp;·&nbsp; Archit Mamodiya &nbsp;·&nbsp; Shagun Vishnoi &nbsp;·&nbsp; Aditya Raj &nbsp;·&nbsp; Naveen Kumar
</div>
<div class="affiliations">
  Department of Computer Science and Engineering · Section A · Team T047
</div>

<div class="abstract-box">
  <div class="abstract-title">Abstract—</div>Financial card fraud detection in production banking environments faces severe class imbalance (~0.17%), delayed verification ground truth (often 30–90 days for customer disputes), and strict constraints on human investigation bandwidth. While supervised classifiers degrade when fraud topologies evolve, unsupervised density estimation provides an operational triage filter by modelling purely legitimate consumer behavior. In this Phase 1 study, we investigate whether a Gaussian Mixture Model (GMM) trained exclusively on normal transactions can effectively rank fraudulent swipes for manual triage under realistic operational constraints. Using the benchmark ULB European cardholder dataset (283,726 deduplicated transactions across 48 hours), we enforce strict temporal integrity via a 60/20/20 chronological split. Feature distributions reject univariate normality (p &lt; 10⁻¹⁰⁰), confirming multimodality. Training exclusively on 169,893 normal transactions from the first 60% of time, we execute an extensive grid search across component counts (K = 1 to 8) and random initialisations. The champion configuration (K = 3, Full Covariance, 29 features) achieves an Average Precision (AP) of 0.612 (95% bootstrap CI [0.485, 0.743]) on the completely unseen test period, outperforming the random-guess baseline (0.0013) by 470×. At an operational false-positive rate budget fixed at ≤ 0.5% on validation (τ = 94.68), the model intercepts 56 of 74 test frauds (75.7% Recall) with only 158 false alarms (0.28% FPR). In human review triage scenarios, the top 50 flagged transactions achieve an alert precision of 84.0% (42 confirmed frauds). Our empirical findings demonstrate that full-covariance mixture models capture subtle multivariate cross-correlations that univariate feature selection discards, establishing an unsupervised baseline for ongoing Phase 2 autoencoder research.
  <div class="keywords"><b>Index Terms</b>—Fraud detection, unsupervised anomaly detection, Gaussian mixture models, expectation-maximization, class imbalance, precision-recall curve, operational triage.</div>
</div>

<div class="columns">

<h1>I. Introduction</h1>
<p><span class="dropcap">C</span>redit card fraud causes tens of billions of dollars in annual global losses, posing an existential challenge to financial payment processors and consumer banking infrastructure [1]. Designing automated fraud defense mechanisms involves three fundamental characteristics of transactional data:</p>
<ol>
  <li><b>Extreme Class Imbalance:</b> Fraudulent activities constitute a minuscule fraction of daily volume (typically under 0.2%). Standard accuracy is uninformative, as a static null classifier predicting all transactions as legitimate achieves &gt;99.8% accuracy while failing completely.</li>
  <li><b>Delayed Verification Ground Truth:</b> In production environments, fraud labels are not immediately available. Fraud is typically reported weeks later via chargebacks, merchant reconciliations, or customer disputes [3]. Supervised classifiers that require contemporaneous labels suffer severe degradation under operational delay.</li>
  <li><b>Analyst Capacity Constraints:</b> Security operations centers have finite human review bandwidth. An algorithm generating thousands of low-confidence alerts overwhelms fraud investigation personnel, leading to alert fatigue and high operational costs.</li>
</ol>
<p>To circumvent the latency of supervised label acquisition, unsupervised anomaly detection provides an operational alternative [4]. By training a density model strictly on abundant, verified historical legitimate purchases, the system establishes a probabilistic boundary of normal cardholder behavior. When incoming transactions exhibit low likelihood under the learned distribution, they receive high anomaly scores and are triaged into an analyst priority queue.</p>
<p>In this work (Phase 1 of Project 26), we develop and empirically validate an unsupervised fraud triage system utilizing Gaussian Mixture Models (GMM). We formulate anomaly detection through negative log-likelihood (NLL) density scoring. We evaluate our pipeline on the ULB credit card benchmark under a time-ordered chronological split, comparing full feature representations against dimensionality-reduced candidate sets.</p>

<h1>II. Related Work</h1>
<p>Machine learning in credit card fraud detection has evolved across multiple paradigms. Dal Pozzolo et al. [1, 2] formalized the benchmark evaluation framework for card fraud, proving that random k-fold cross-validation introduces severe temporal lookahead leakage and inflates performance metrics. They established that chronological partitioning and human verification capacity metrics (Precision@k) are mandatory for realistic assessment.</p>
<p>Unsupervised anomaly detection has been surveyed comprehensively by Chandola et al. [4]. Classical methods include distance-based algorithms, linear projections, and parametric density models. Gaussian Mixture Models, optimized via Expectation-Maximization (EM) [5], offer analytical density evaluations and capture multimodal consumer spending habits through heterogeneous component centers and full covariance matrices.</p>
<p>Davis and Goadrich [6] established that under massive class skew, the Receiver Operating Characteristic (ROC) curve is structurally flattered by the overwhelming volume of True Negatives. They demonstrated that Precision-Recall (PR) curves and Average Precision (AP) provide an uncompromised assessment of minority-class retrieval.</p>
<p>Recent advancements explore deep autoencoding architectures [10] and deep autoencoding Gaussian mixture models (DAGMM) [7], which unify dimensionality reduction and density estimation. Our Phase 1 GMM baseline establishes the mathematical benchmark against which Phase 2 autoencoders will be contrasted.</p>

<h1>III. Dataset and Time Partitioning</h1>
<h2>A. Dataset Properties</h2>
<p>We utilize the public European cardholder benchmark dataset collected by the Machine Learning Group at Université Libre de Bruxelles (ULB) [2]. The dataset contains 284,807 transactions recorded over a continuous 48-hour period in September 2013, with 492 confirmed fraudulent events (0.173% prevalence).</p>
<p>The dataset consists of 30 numerical input features: <code>Time</code> (seconds from first transaction), <code>Amount</code> (in Euros), <code>V1</code>–<code>V28</code> (anonymized continuous numerical attributes derived from a proprietary Principal Component Analysis performed by the banking institution for data privacy), and <code>Class</code> (binary ground truth: 0 = Normal, 1 = Fraud).</p>

<h2>B. Deduplication and Chronological Partitioning</h2>
<p>Data auditing identified 1,081 exact duplicate rows (including 19 fraudulent records), which were purged to prevent artificial density inflation, leaving 283,726 clean rows.</p>

<div class="table-caption">Table I: Chronological Data Partitioning (60 / 20 / 20 Split)</div>
<table class="booktabs">
  <thead>
    <tr>
      <th>Split</th>
      <th>Window</th>
      <th>Rows</th>
      <th>Frauds</th>
      <th>Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Train</td>
      <td>First 60% (0–28.8 h)</td>
      <td>169,893</td>
      <td>0 (342 rem.)</td>
      <td>0.000%</td>
    </tr>
    <tr>
      <td>Validation</td>
      <td>Next 20% (28.8–38.4 h)</td>
      <td>56,745</td>
      <td>57</td>
      <td>0.100%</td>
    </tr>
    <tr>
      <td>Test</td>
      <td>Last 20% (38.4–48.0 h)</td>
      <td>56,746</td>
      <td>74</td>
      <td>0.130%</td>
    </tr>
  </tbody>
</table>

<p>To prevent temporal data leakage, we enforce a strict 60/20/20 chronological partition based on <code>Time</code> (Table I):</p>
<ol>
  <li><b>Training (First 60%):</b> 170,235 transactions. All 342 fraud records occurring in this window are removed. The remaining 169,893 purely legitimate transactions are used to fit the scaler and the GMM.</li>
  <li><b>Validation (Next 20%):</b> 56,745 transactions (57 frauds, 0.100% rate). Used to optimize hyperparameters (K, covariance constraints, random seeds) and tune the operational decision threshold.</li>
  <li><b>Test (Final 20%):</b> 56,746 transactions (74 frauds, 0.130% rate). Kept strictly sealed until model finalization for out-of-time evaluation.</li>
</ol>

<h1>IV. Exploratory Data Analysis</h1>
<p>All exploratory analysis and feature ranking are conducted exclusively on the training period without peeking at validation or test labels.</p>

<h2>A. Diurnal Transaction Distributions</h2>
<p>We derive transaction hour as: <i>Hour = ⌊Time / 3600⌋ mod 24</i>. As illustrated in Fig. 1, normal transaction volume exhibits strong diurnal rhythm, dropping significantly between 01:00 and 06:00. Conversely, fraudulent swipes remain active during nocturnal hours, spiking between 02:00 and 04:00 where detection friction is minimal.</p>

<div class="figure-container">
  <img src="../results/figures/fig1_time_diurnal_distribution.png" alt="Fig 1: Diurnal Distribution">
  <div class="figure-caption"><b>Fig. 1.</b> Diurnal transaction volume distribution (Normal vs. Fraud) during the training period, demonstrating nocturnal fraud concentration.</div>
</div>

<h2>B. Amount Log-Transformation</h2>
<p>Raw purchase amounts span from €0.00 to €25,691.16 with a median of €22.00, yielding extreme positive skewness (γ₁ = 15.77). To accommodate the Gaussian structural assumptions of our mixture components, we apply a natural logarithmic transformation: <i>Amount̃ = ln(1 + Amount)</i>.</p>
<p>Fig. 2 verifies that the transformation reduces skewness to γ₁ = 0.16, yielding an approximately symmetric bell-shaped distribution suitable for GMM covariance estimation.</p>

<div class="figure-container">
  <img src="../results/figures/fig2_amount_distribution_log.png" alt="Fig 2: Amount Distribution">
  <div class="figure-caption"><b>Fig. 2.</b> Distribution of transaction Amount: Raw distribution exhibits extreme right-skew (γ₁ = 15.77), while log(1 + Amount) normalizes the profile (γ₁ = 0.16).</div>
</div>

<h2>C. Feature Divergence Analysis (KS Test)</h2>
<p>To identify the most discriminative features, we compute the two-sample Kolmogorov-Smirnov (KS) statistic [9] between normal and fraudulent training distributions: <i>D_KS = sup |F_normal(x) - F_fraud(x)|</i>.</p>
<p>The top 11 divergent features (V14, V10, V12, V11, V17, V4, V3, V16, V7, V2, V9) all exhibit D_KS &gt; 0.32 (Fig. 3). These 11 features, combined with Amount̃, form our candidate 12-feature subset recorded in <code>configs/gmm_features.json</code>.</p>

<div class="figure-container">
  <img src="../results/figures/fig3_top_divergent_features_kde.png" alt="Fig 3: KDE of Top Features">
  <div class="figure-caption"><b>Fig. 3.</b> Kernel density estimations for the top six separating features identified by the two-sample Kolmogorov-Smirnov test.</div>
</div>

<div class="figure-container">
  <img src="../results/figures/fig4_feature_correlation_matrix.png" alt="Fig 4: Feature Correlation Matrix">
  <div class="figure-caption"><b>Fig. 4.</b> Correlation matrix of Hour, Amount, and top separating PCA features.</div>
</div>

<h2>D. Multimodality and Normality Rejection</h2>
<p>We applied the D'Agostino-Pearson omnibus test [8] on legitimate transactions for Amount, V1, V4, and V14. Every tested feature yielded p-values &lt; 10⁻¹⁰⁰, rejecting unimodal Gaussian behavior. Legitimate consumer spending consists of distinct behavioral personas (e.g., micro-purchases, recurring utilities, commercial corporate cards), necessitating a multi-component mixture model.</p>

<h1>V. Methodology: GMM Density Scoring</h1>
<h2>A. Mathematical Formulation</h2>
<p>A Gaussian Mixture Model models the probability density function of continuous feature vector x ∈ ℝᴰ as a convex combination of K multivariate Gaussian distributions:</p>
<div class="math-block">p(x | θ) = ∑_{k=1}^K π_k N(x | μ_k, Σ_k)</div>
<p class="no-indent">subject to mixing proportions ∑ π_k = 1 (π_k ≥ 0), where component density is defined as:</p>
<div class="math-block">N(x | μ_k, Σ_k) = (2π)^(-D/2) |Σ_k|^(-1/2) exp(-½ (x - μ_k)ᵀ Σ_k⁻¹ (x - μ_k))</div>
<p>Model parameters θ = {π_k, μ_k, Σ_k} are estimated on unlabelled legitimate training data via the Expectation-Maximization (EM) algorithm [5] with regularisation covariance ε = 10⁻⁵ added to diagonals to prevent numerical collapse.</p>

<h2>B. Anomaly Scoring Function</h2>
<p>Transactions residing in low-density probability regions reflect anomalous behavioral deviations. We compute the anomaly score S(x) as the Negative Log-Likelihood (NLL):</p>
<div class="math-block">S(x) = -ln p(x | θ̂) = -ln [ ∑_{k=1}^K π̂_k N(x | μ̂_k, Σ̂_k) ]</div>
<p>High values of S(x) trigger security triage alerts.</p>

<h2>C. Hyperparameter Grid Protocol</h2>
<p>The search space evaluated: Feature sets (All 29 vs Selected 12), Components K ∈ {1, 2, ..., 8}, and 3 random seeds ({0, 1, 2}) with n_init = 5 restarts (144 fits total). Models were selected based on validation Average Precision (AP), with Bayesian Information Criterion (BIC) monitored for complexity penalization.</p>

<div class="figure-container">
  <img src="../results/figures/fig5_gmm_component_selection.png" alt="Fig 5: Component Selection">
  <div class="figure-caption"><b>Fig. 5.</b> Hyperparameter search across component count K: (Left) BIC trajectories; (Right) Validation Average Precision across random seeds.</div>
</div>

<div class="table-caption">Table II: Validation Performance Across Configurations</div>
<table class="booktabs">
  <thead>
    <tr>
      <th>Features</th>
      <th>K</th>
      <th>AP Mean</th>
      <th>AP Min</th>
      <th>AP Max</th>
      <th>Mean BIC</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>All 29</td><td>1</td><td>0.074</td><td>0.074</td><td>0.074</td><td>13,701,380</td></tr>
    <tr><td>All 29</td><td>2</td><td>0.176</td><td>0.079</td><td>0.371</td><td>6,635,476</td></tr>
    <tr class="highlight"><td>All 29</td><td>3</td><td>0.549</td><td>0.343</td><td>0.666</td><td>3,412,903</td></tr>
    <tr><td>All 29</td><td>4</td><td>0.068</td><td>0.065</td><td>0.074</td><td>-1,218,040</td></tr>
    <tr><td>All 29</td><td>5</td><td>0.300</td><td>0.058</td><td>0.647</td><td>-2,666,021</td></tr>
    <tr><td>All 29</td><td>6</td><td>0.068</td><td>0.059</td><td>0.078</td><td>-3,176,066</td></tr>
    <tr><td>All 29</td><td>7</td><td>0.196</td><td>0.128</td><td>0.232</td><td>-3,987,216</td></tr>
    <tr><td>All 29</td><td>8</td><td>0.323</td><td>0.246</td><td>0.419</td><td>-5,122,966</td></tr>
    <tr><td>Sel. 12</td><td>1</td><td>0.266</td><td>0.266</td><td>0.266</td><td>5,643,466</td></tr>
    <tr><td>Sel. 12</td><td>2</td><td>0.205</td><td>0.201</td><td>0.213</td><td>4,670,746</td></tr>
    <tr><td>Sel. 12</td><td>3</td><td>0.201</td><td>0.199</td><td>0.205</td><td>4,387,461</td></tr>
    <tr><td>Sel. 12</td><td>4</td><td>0.197</td><td>0.197</td><td>0.198</td><td>4,047,104</td></tr>
    <tr><td>Sel. 12</td><td>5</td><td>0.203</td><td>0.198</td><td>0.212</td><td>3,833,669</td></tr>
    <tr><td>Sel. 12</td><td>6</td><td>0.334</td><td>0.081</td><td>0.628</td><td>3,628,444</td></tr>
    <tr><td>Sel. 12</td><td>7</td><td>0.067</td><td>0.065</td><td>0.069</td><td>3,297,546</td></tr>
    <tr><td>Sel. 12</td><td>8</td><td>0.075</td><td>0.065</td><td>0.095</td><td>3,158,924</td></tr>
  </tbody>
</table>

<h1>VI. Experimental Results and Discussion</h1>
<h2>A. Validation Findings</h2>
<p>As summarized in Table II and Fig. 5:</p>
<ol>
  <li><b>Failure of BIC for Fraud Ranking:</b> For All 29 features, BIC turns negative for K ≥ 4 as Gaussian components collapse onto localized data pockets, inflating likelihood without improving fraud ranking. Selection must rely on validation AP.</li>
  <li><b>Dimensionality vs. Flexibility Trade-off:</b> At K=1, Selected 12 features outperform All 29 (AP = 0.266 vs 0.074). However, at K=3, All 29 reaches the study's peak (AP_mean = 0.549, peak seed AP = 0.6661). The 29×29 full covariance matrix captures 406 pairwise relationships that univariate ranking discarded.</li>
  <li><b>EM Solution Variance:</b> GMM quality varies substantially with initialization seed. For All 29 at K=3, AP ranges from 0.343 to 0.666, demonstrating the necessity of multi-seed evaluations.</li>
</ol>
<p><b>Champion Model:</b> All 29 features, K = 3, Full Covariance, Seed 2 (Validation AP = 0.6661).</p>

<h2>B. Operational Decision Thresholding</h2>
<p>The operational threshold τ was calibrated on validation under Scenario A (False Alarm Budget): <i>τ = Percentile_99.5(S_normal) = 94.68</i>, guaranteeing that ≥99.5% of legitimate cardholders experience uninterrupted service (FPR ≤ 0.5%).</p>

<div class="figure-container">
  <img src="../results/figures/fig6_gmm_test_pr_curve.png" alt="Fig 6: PR Curve">
  <div class="figure-caption"><b>Fig. 6.</b> Precision-Recall Curve on the held-out test set (AP = 0.6121), operating at 470× the random baseline (0.13%).</div>
</div>

<div class="figure-container">
  <img src="../results/figures/fig7_gmm_test_confusion_matrix.png" alt="Fig 7: Confusion Matrix">
  <div class="figure-caption"><b>Fig. 7.</b> Test Confusion Matrix evaluated at the frozen validation threshold τ = 94.68 (0.5% False Alarm Budget).</div>
</div>

<h2>C. Out-of-Time Test Set Evaluation</h2>
<p>Applying the frozen champion GMM and threshold τ = 94.68 to the 56,746 untouched test transactions yielded the results in Table III.</p>

<div class="table-caption">Table III: Test Performance and 95% Bootstrap CIs</div>
<table class="booktabs">
  <thead>
    <tr>
      <th>Metric</th>
      <th>Test Value</th>
      <th>95% Bootstrap CI</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Random Baseline AP</td><td>0.0013 (0.13%)</td><td>—</td></tr>
    <tr class="highlight"><td>Average Precision (AP)</td><td><b>0.6121</b></td><td>[0.485, 0.743]</td></tr>
    <tr><td>PR-AUC (Trapezoidal)</td><td>0.6083</td><td>—</td></tr>
    <tr><td>ROC-AUC</td><td><b>0.9229</b></td><td>[0.879, 0.962]</td></tr>
    <tr class="highlight"><td>Recall @ τ = 94.68</td><td><b>75.68%</b> (56/74)</td><td>[65.79%, 84.72%]</td></tr>
    <tr><td>Realized False Alarm Rate</td><td><b>0.28%</b> (158/56,672)</td><td>—</td></tr>
    <tr><td>Precision @ Top 25 Alerts</td><td><b>72.00%</b> (18/25)</td><td>—</td></tr>
    <tr class="highlight"><td>Precision @ Top 50 Alerts</td><td><b>84.00%</b> (42/50)</td><td>[70.00%, 94.00%]</td></tr>
    <tr><td>Precision @ Top 100 Alerts</td><td><b>56.00%</b> (56/100)</td><td>—</td></tr>
  </tbody>
</table>

<p><b>Generalization:</b> Test AP (0.6121) closely tracks validation AP (0.6661) and falls well within the bootstrap interval [0.485, 0.743], confirming strong generalization without overfitting across time boundaries.</p>
<p><b>Threshold Verification:</b> At τ = 94.68, the model catches 56 of 74 frauds (75.7% Recall) with only 158 false alarms (0.28% FPR), safely satisfying the bank's 0.5% budget (Fig. 7).</p>
<p><b>Triage Efficiency:</b> In human review queues, the top 50 flagged swipes achieve 84.0% precision (42 confirmed frauds), minimizing investigation waste. Precision moderates to 56.0% at Top 100 as the finite pool of 74 frauds is exhausted.</p>

<h1>VII. Conclusion & Phase 2 Roadmap</h1>
<p>This Phase 1 study establishes that unsupervised Gaussian Mixture Models provide a viable frontline triage filter for credit card fraud. By scoring transactions via negative log-likelihood under legitimate data, our pipeline achieved AP = 0.6121 (470× random guess), captured 75.7% of out-of-time fraud at 0.28% FPR, and achieved 84% precision in top-50 review queues.</p>
<p><b>Phase 2 Transition:</b> In Phase 2 (Deep Learning), we will implement deep autoencoder architectures to benchmark reconstruction error against GMM density scoring, conduct ablation studies, and investigate hybrid autoencoder-GMM representations.</p>

<div class="references">
<h1>References</h1>
<ol>
  <li>A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and evaluation framework," <i>IEEE Trans. Neural Netw. Learn. Syst.</i>, vol. 29, no. 8, pp. 3784–3796, 2018.</li>
  <li>A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating probability with undersampling for unbalanced classification," in <i>IEEE SSCI</i>, 2015, pp. 159–166.</li>
  <li>F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer, and G. Bontempi, "SCARFF: A framework for addressing drift in fraud detection with streaming active learning," <i>Inf. Sci.</i>, vol. 465, pp. 252–270, 2018.</li>
  <li>V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," <i>ACM Comput. Surv.</i>, vol. 41, no. 3, pp. 1–58, 2009.</li>
  <li>C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>. New York: Springer, 2006.</li>
  <li>J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in <i>Proc. 23rd ICML</i>, 2006, pp. 233–240.</li>
  <li>B. Zong et al., "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in <i>ICLR</i>, 2018.</li>
  <li>R. D'Agostino and E. S. Pearson, "Tests for departure from normality," <i>Biometrika</i>, vol. 60, no. 3, pp. 613–622, 1973.</li>
  <li>F. J. Massey, "The Kolmogorov-Smirnov test for goodness of fit," <i>J. Am. Stat. Assoc.</i>, vol. 46, no. 253, pp. 68–78, 1951.</li>
  <li>M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in <i>Proc. MLSDA</i>, 2014, pp. 4–11.</li>
</ol>
</div>

</div>

</body>
</html>
"""

    html_path = "reports/Card_Fraud_Triage_Phase1_Report.html"
    pdf_path = "reports/Card_Fraud_Triage_Phase1_Report.pdf"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML report generated at {html_path}")

    # Compile HTML to PDF using Chrome Headless
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_path
    ]
    subprocess.run(chrome_cmd, check=True)
    print(f"PDF generated successfully at {pdf_path} (Size: {os.path.getsize(pdf_path):,} bytes)")

    # Render PDF pages to high-res PNG previews using pypdfium2
    pdf = pdfium.PdfDocument(pdf_path)
    preview_paths = []
    for i, page in enumerate(pdf):
        image = page.render(scale=2.0).to_pil()
        preview_file = f"reports/preview_page_{i+1}.png"
        image.save(preview_file)
        preview_paths.append(preview_file)
        print(f"Rendered preview image: {preview_file} ({image.size[0]}x{image.size[1]})")

    return pdf_path, preview_paths

if __name__ == "__main__":
    generate_report()
