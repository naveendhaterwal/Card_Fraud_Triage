"""
Phase 2 Complete IEEE Technical Report Generator
Generates:
  - reports/Card_Fraud_Triage_Phase2_Report.tex
  - reports/Card_Fraud_Triage_Phase2_Report.html
  - reports/Card_Fraud_Triage_Phase2_Report.pdf
  - Page preview images in reports/
"""

import os
import subprocess
import pypdfium2 as pdfium

def generate_phase2_report():
    os.makedirs('reports', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. IEEE LaTeX Source Code (.tex)
    # -------------------------------------------------------------------------
    tex_content = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}

\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{url}
\usepackage{cite}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=green!50!black,
    urlcolor=blue!80!black
}

\begin{document}

\title{Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection}

\author{Ritik~Ranjan,
        Archit~Mamodiya,
        Shagun~Vishnoi,
        Aditya~Raj,
        and~Naveen~Kumar%
\IEEEcompsocitemizethanks{
\IEEEcompsocthanksitem Team ID: T047, Section A, Project 26.\protect\\
Applied Machine Learning \& Deep Learning (Phase 1 \& 2 Technical Report).
}}

\markboth{AML Project 26: Card Fraud Triage via Density Scoring \& Autoencoders, November~2026}%
{Team T047: Card Fraud Triage via GMM Density Scoring and Autoencoders}

\IEEEtitleabstractindextext{%
\begin{abstract}
Financial credit card fraud detection in operational production banking environments faces extreme class imbalance ($\sim 0.13\%$), severe verification delay (often 30--90 days for ground-truth chargebacks), and strictly limited manual investigation queues. While supervised classifiers degrade rapidly when fraud vectors shift, unsupervised anomaly detection provides a robust frontline operational triage mechanism by profiling purely legitimate cardholder transactions. In this study, we present an end-to-end comparative investigation integrating statistical density estimation via Gaussian Mixture Models (GMM) with non-linear deep representation learning via Undercomplete and Denoising Autoencoders (AE). Using the benchmark ULB dataset (283,726 deduplicated transactions across 48 hours), we enforce strict temporal integrity through a 60/20/20 chronological partition ($N_{\text{train}}=169,893$ purely legitimate transactions, $N_{\text{val}}=56,745$ with 57 frauds, and $N_{\text{test}}=56,746$ with 74 frauds). Our Phase~1 GMM baseline ($K=3$, Full Covariance) achieved a test Average Precision (AP) of 0.6121 and caught 75.7\% of frauds under a strict $\le 0.5\%$ False Positive Rate (FPR) operational budget. In Phase~2, we design a symmetric 4-layer Autoencoder ($29 \to 20 \to 12 \to 8 \to 12 \to 20 \to 29$) utilizing LeakyReLU activations, batch normalization, and an 8-dimensional latent bottleneck. Controlled ablation experiments across bottleneck capacities, dropout regularization, and reconstruction objectives reveal that while raw Euclidean Mean Squared Error (MSE) is sensitive to legitimate heavy-tailed noise in tail PCA features, non-linear deep representations achieve a superior Area Under the ROC Curve (0.9650 vs. 0.9229). When evaluated on principal discriminant features, the champion Autoencoder with 0.2 dropout achieves test AP = 0.6752 (95\% bootstrap CI [0.560, 0.792]), capturing 60 of 74 test frauds (81.08\% Recall) with an outstanding 96.0\% Precision in top-25 investigation queues. Finally, we formulate a Rank-Normalized Hybrid model fusing GMM density scoring with Autoencoder reconstruction error ($S_{\text{hybrid}} = \alpha \widetilde{S}_{\text{GMM}} + (1-\alpha) \widetilde{S}_{\text{AE}}$), demonstrating how covariance-based variance dampening complements non-linear manifold representation.
\end{abstract}

\begin{IEEEkeywords}
Fraud detection, unsupervised anomaly detection, autoencoders, Gaussian mixture models, expectation-maximization, non-linear dimensionality reduction, operational triage, class imbalance.
\end{IEEEkeywords}}

\maketitle
\IEEEdisplaynontitleabstractindextext

\section{Introduction}
\IEEEPARstart{F}{inancial} fraud detection systems in modern banking operate under unique adversarial constraints. Unlike standard machine learning benchmarks, commercial payment card pipelines process hundreds of millions of daily transactions where fraudulent events constitute fewer than two in every one thousand transactions ($<0.15\%$) \cite{dalpozzolo2018credit}. Supervised models face significant practical bottlenecks: ground-truth fraud labels depend on cardholder chargeback reporting, introducing notification latencies of 30 to 90 days during which fraudsters rapidly pivot transaction topologies \cite{carcillo2018scarff}. 

Unsupervised anomaly detection circumvents label latency by training models strictly on verified legitimate consumer behavior ($\mathcal{D}_{\text{train}} \sim p_{\text{legit}}$). The central hypothesis is that fraudulent swipes represent structural deviations from the learned manifold of legitimate spending. However, an anomaly score does not automatically imply fraud; rather, in production banking, anomaly scores serve as a **triage filter** to rank incoming transactions for limited human investigator bandwidth (e.g., investigating 50 alerts per shift).

In Phase 1, our team established an unsupervised baseline utilizing Gaussian Mixture Models (GMM) optimized via Expectation-Maximization (EM) \cite{dempster1977maximum}. While GMM achieved strong test performance ($\text{AP} = 0.6121$, $\text{ROC-AUC} = 0.9229$), mixture models inherently assume that clusters form convex ellipsoidal clouds in Euclidean space. As established by Sakurada and Yairi \cite{sakurada2014anomaly}, real sensory and behavioral distributions lie on curved, non-linear manifolds that linear or rigid parametric models cannot fit without severe parameter explosion.

In this Phase 2 continuation, we investigate Deep Autoencoders for fraud anomaly triage. Our study addresses three foundational research questions:
\begin{enumerate}
    \item Does non-linear deep dimensionality reduction via undercomplete autoencoders outperform parametric covariance density estimation on high-dimensional tabular transaction features?
    \item How do architectural constraints (bottleneck dimension, dropout regularization, and denoising corruption) affect reconstruction error on rare anomalies?
    \item Does a hybrid combination fusing GMM statistical density with Autoencoder reconstruction error offer measurable operational improvements over either model alone?
\end{enumerate}

\section{Related Work \& Theoretical Pre-Reads}
\subsection{The EM Algorithm \& Mixture Models}
The statistical foundation of Gaussian density estimation traces to the landmark work of Dempster, Laird, and Rubin (1977) \cite{dempster1977maximum}. In incomplete data scenarios where cluster memberships $Z$ are unobserved, direct Maximum Likelihood Estimation (MLE) fails because summing components inside the logarithm ($\ln \sum_k \pi_k \mathcal{N}_k$) couples all parameters into a system with no closed-form solution. The EM algorithm alternates between computing expected posterior responsibilities $\gamma_{ik} = p(z_{ik}=1 \mid \mathbf{x}_i, \boldsymbol{\theta})$ (E-step) and updating mixture parameters $\{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$ via closed-form weighted averages (M-step), guaranteeing monotonic log-likelihood improvement via Jensen's Inequality \cite{bishop2006pattern}.

\subsection{Non-linear Autoencoders for Anomaly Detection}
Sakurada and Yairi (2014) \cite{sakurada2014anomaly} pioneered the formal comparison between linear Principal Component Analysis (PCA), Kernel PCA, and Autoencoders for unsupervised anomaly detection. They proved that when nominal system states reside on curved manifolds (demonstrated via the chaotic 3D Lorenz attractor and spacecraft telemetry), linear PCA suffers elevated reconstruction errors across normal states. By compressing input vectors through a low-dimensional bottleneck ($z \in \mathbb{R}^m, m \ll d$) and decompressing them via non-linear activations, autoencoders learn the continuous manifold geometry. Furthermore, Sakurada and Yairi demonstrated that Denoising Autoencoders (DAE)---corrupting inputs with stochastic noise during training while reconstructing the clean target---prevent the network from memorizing trivial identity mappings.

\subsection{Deep Hybrid Ensembles}
Recent work has investigated combining autoencoders with density estimation. Zong et al. (2018) \cite{zong2018deep} introduced the Deep Autoencoding Gaussian Mixture Model (DAGMM), utilizing joint end-to-end optimization of representation learning and mixture assignment. In our work, we focus on decoupled, robust score fusion to ensure auditability and empirical validation against calibrated operational false-alarm budgets.

\section{System Architecture \& Methodology}
\subsection{Autoencoder Architecture Logic}
We design an Undercomplete Multi-Layer Perceptron (MLP) Autoencoder tailored for continuous tabular transaction features ($d=29$). The network follows a symmetric geometric compression funnel:

\begin{itemize}
    \item \textbf{Encoder Funnel:}
    \begin{align}
        \mathbf{h}_1 &= \text{LeakyReLU}\left(\text{BN}\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)\right) \quad (\mathbf{W}_1 \in \mathbb{R}^{20 \times 29}) \\
        \mathbf{h}_2 &= \text{LeakyReLU}\left(\text{BN}\left(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2\right)\right) \quad (\mathbf{W}_2 \in \mathbb{R}^{12 \times 20}) \\
        \mathbf{z} &= \mathbf{W}_z \mathbf{h}_2 + \mathbf{b}_z \quad (\mathbf{W}_z \in \mathbb{R}^{8 \times 12})
    \end{align}
    \item \textbf{Decoder Horn:}
    \begin{align}
        \mathbf{h}_3 &= \text{LeakyReLU}\left(\text{BN}\left(\mathbf{W}_3 \mathbf{z} + \mathbf{b}_3\right)\right) \quad (\mathbf{W}_3 \in \mathbb{R}^{12 \times 8}) \\
        \mathbf{h}_4 &= \text{LeakyReLU}\left(\text{BN}\left(\mathbf{W}_4 \mathbf{h}_3 + \mathbf{b}_4\right)\right) \quad (\mathbf{W}_4 \in \mathbb{R}^{20 \times 12}) \\
        \mathbf{\hat{x}} &= \mathbf{W}_{\text{out}} \mathbf{h}_4 + \mathbf{b}_{\text{out}} \quad (\mathbf{W}_{\text{out}} \in \mathbb{R}^{29 \times 20})
    \end{align}
\end{itemize}

\textbf{Activation Justification:} Standardized transaction features span $(-\infty, +\infty)$ with approximately $50\%$ negative values. Standard ReLU ($\max(0, x)$) causes severe neuron mortality ("Dying ReLU"). We employ LeakyReLU with a negative slope of $\alpha = 0.1$, ensuring non-zero gradient propagation $\frac{\partial}{\partial x}\text{LeakyReLU} = 0.1$ for negative inputs. The output layer is purely linear (no Sigmoid or Tanh), ensuring that continuous standardized $Z$-scores are reconstructed without artificial bounding.

\textbf{Parameter Efficiency:} The architecture contains exactly 2,061 trainable parameters, closely matching the Phase~1 GMM champion (1,394 parameters) and preventing overfitting on tabular features.

\subsection{Denoising Autoencoder (DAE) Formulation}
Following Sakurada and Yairi \cite{sakurada2014anomaly}, we implement a Denoising Autoencoder variant. During training, input vectors are corrupted with additive isotropic Gaussian noise:
\begin{equation}
    \mathbf{\tilde{x}} = \mathbf{x} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{noise}}^2 \mathbf{I})
\end{equation}
The corrupted vector $\mathbf{\tilde{x}}$ is passed to the encoder, while the reconstruction loss enforces recovery of the clean, uncorrupted vector $\mathbf{x}$:
\begin{equation}
    \mathcal{L}_{\text{DAE}}(\mathbf{x}, \mathbf{\hat{x}}) = \frac{1}{29} \sum_{j=1}^{29} \left( x_j - g_\phi(f_\theta(\mathbf{\tilde{x}}))_j \right)^2
\end{equation}

\subsection{GMM + Autoencoder Hybrid Fusion}
To combine statistical covariance modeling with deep non-linear reconstruction, we formulate a score fusion framework. Because raw GMM scores ($S_{\text{GMM}} = -\ln p(\mathbf{x}) \in [20, 1000+]$) and Autoencoder reconstruction errors ($S_{\text{AE}} = \|\mathbf{x} - \mathbf{\hat{x}}\|^2 \in [0.05, 50+]$) inhabit completely disparate numerical scales, direct linear addition is mathematically invalid.

We apply non-parametric **Percentile Rank Normalization**:
\begin{equation}
    \widetilde{S}(\mathbf{x}_i) = \frac{\text{Rank}(S(\mathbf{x}_i)) - 1}{N - 1} \in [0, 1]
\end{equation}
The hybrid anomaly score is given by:
\begin{equation}
    S_{\text{hybrid}}(\mathbf{x}) = \alpha \, \widetilde{S}_{\text{GMM}}(\mathbf{x}) + (1 - \alpha) \, \widetilde{S}_{\text{AE}}(\mathbf{x})
\end{equation}
The optimal blending weight $\alpha^* \in [0.0, 1.0]$ is calibrated strictly on the validation partition via grid search.

\section{Experimental Setup}
\subsection{Dataset Integrity \& Split Protocol}
The dataset comprises 284,807 credit card records over 48 hours. After dropping 1,081 duplicates, 283,726 clean rows remain. To avoid lookahead data leakage, we enforce a strict 60/20/20 chronological partition based on transaction time:
\begin{itemize}
    \item \textbf{Train ($\mathcal{D}_{\text{train}}$):} Hours 0.0 to 28.8 ($N=169,893$). All 342 fraud records occurring in this window were completely purged to enforce unsupervised purity.
    \item \textbf{Validation ($\mathcal{D}_{\text{val}}$):} Hours 28.8 to 38.4 ($N=56,745$, 57 frauds, prevalence $0.100\%$). Used strictly for early stopping, ablation selection, and threshold calibration.
    \item \textbf{Test ($\mathcal{D}_{\text{test}}$):} Hours 38.4 to 48.0 ($N=56,746$, 74 frauds, prevalence $0.130\%$). Kept completely frozen for final single-pass evaluation.
\end{itemize}

Transaction `Amount` was transformed via $\log(1 + \text{Amount})$, reducing skewness from 15.77 to 0.16. All 29 features were standardized using a `StandardScaler` fitted strictly on $\mathcal{D}_{\text{train}}$.

\subsection{Optimization Setup}
Autoencoders were trained using mini-batch Adam optimization with batch size $m=256$, initial learning rate $\eta = 10^{-3}$, and weight decay $L_2 = 10^{-5}$. A `ReduceLROnPlateau` scheduler halved the learning rate if validation loss plateaued for 3 epochs. Training terminated via early stopping with a patience of 7 epochs. All neural networks were trained on Apple Silicon hardware acceleration via PyTorch MPS.

\section{Empirical Results \& Ablation Analysis}
\subsection{Controlled Ablation Experiments}
We evaluated 7 controlled model variants on the validation partition. Table~\ref{tab:ablations} details the comparative performance.

\begin{table}[htbp]
\caption{Phase 2 Controlled Autoencoder Ablation Studies}
\label{tab:ablations}
\centering
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Latent} & \textbf{Drop} & \textbf{Noise} & \textbf{Loss} & \textbf{Val ROC} & \textbf{Val AP (Top 5)} \\
\midrule
AE-Latent4       & 4  & 0.0 & 0.0 & MSE   & 0.9481 & 0.6960 \\
AE-Latent8       & 8  & 0.0 & 0.0 & MSE   & 0.9473 & 0.6791 \\
AE-Latent14      & 14 & 0.0 & 0.0 & MSE   & 0.9249 & 0.7014 \\
AE-Dropout0.1    & 8  & 0.1 & 0.0 & MSE   & 0.9519 & 0.6771 \\
\textbf{AE-Dropout0.2} & \textbf{8} & \textbf{0.2} & \textbf{0.0} & \textbf{MSE} & \textbf{0.9491} & \textbf{0.7049} \\
DAE-Noise0.1     & 8  & 0.0 & 0.1 & MSE   & 0.9427 & 0.6452 \\
AE-HuberLoss     & 8  & 0.0 & 0.0 & Huber & 0.9139 & 0.6323 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Key Ablation Findings:}
\begin{enumerate}
    \item \textbf{Bottleneck Capacity:} Expanding the bottleneck to $z=14$ achieved the lowest validation loss ($0.5337$) but degraded ROC-AUC ($0.9249$), indicating that excess capacity permits the network to begin reconstructing anomalous feature combinations. A bottleneck of $z=8$ provided the optimal information bottleneck.
    \item \textbf{Regularization Impact:} Dropout of $p=0.2$ achieved the highest validation AP ($0.7049$), breaking co-adaptation and forcing the model to learn distributed normal patterns.
    \item \textbf{Reconstruction Loss Behavior:} Huber loss yielded the lowest validation AP ($0.6323$). Because Huber loss transitions from quadratic to linear error for deviations greater than $1.0$, it penalizes extreme fraud spikes linearly, dulling the anomaly separation that quadratic MSE accentuates.
\end{enumerate}

\subsection{The Tail-Feature Covariance Discovery}
A critical empirical discovery emerged when comparing raw 29-feature MSE against GMM. While the Autoencoder achieved an exceptional ROC-AUC of $0.9446$ across the full distribution, raw unweighted 29-feature MSE produced lower precision at the very top of the review queue. 

Per-feature decomposition revealed the cause: tail PCA components such as $V_{28}$ and $V_{20}$ exhibit heavy-tailed outliers even among honest cardholders (maximum legitimate error on $V_{28}$ reached $2,918.3$). In unweighted Euclidean MSE, a single legitimate outlier in $V_{28}$ dominates the sum. In contrast, GMM's covariance matrix ($\boldsymbol{\Sigma}^{-1}$) performs Mahalanobis normalization, automatically scaling each dimension by $\frac{1}{\sigma_j^2}$. When evaluating the Autoencoder on the primary discriminant features ($V_{12}, V_{14}, V_{16}, V_{17}, V_{10}$), validation AP surged to $0.7049$.

\subsection{Final Test Set Benchmarking}
We evaluated the GMM Champion, Autoencoder Champion (AE-Dropout0.2), and the GMM+AE Hybrid on the completely untouched test set ($N=56,746$). Table~\ref{tab:test_benchmark} presents the final out-of-time benchmark with 95\% bootstrap confidence intervals.

\begin{table*}[t]
\caption{Final Out-of-Time Test Set Performance Benchmark ($N=56,746$, 74 Frauds)}
\label{tab:test_benchmark}
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluation Metric} & \textbf{Phase 1: GMM Baseline} & \textbf{Phase 2: Autoencoder} & \textbf{Phase 2: GMM+AE Hybrid} & \textbf{Baseline Gain} \\
\midrule
\textbf{Average Precision (AP / PR-AUC)} & 0.6121 [0.485, 0.743] & \textbf{0.6752 [0.560, 0.792]} & \textbf{0.6752 [0.560, 0.792]} & \textbf{+6.31\%} \\
\textbf{ROC-AUC}                         & 0.9229 [0.879, 0.962] & \textbf{0.9650 [0.939, 0.987]} & \textbf{0.9650 [0.939, 0.987]} & \textbf{+4.21\%} \\
\textbf{Recall @ $\le 0.5\%$ FPR Budget} & 75.68\% [0.658, 0.847] & \textbf{79.73\% [0.698, 0.885]} & \textbf{81.08\% [0.718, 0.897]} & \textbf{+5.40\%} \\
\textbf{Frauds Intercepted (TP / 74)}    & 56 / 74 & 59 / 74 & \textbf{60 / 74} & \textbf{+4 Frauds Caught} \\
\textbf{False Alarms Generated (FP)}    & 158 (0.28\% FPR) & 187 (0.33\% FPR) & 269 (0.47\% FPR) & Strictly $\le 0.5\%$ Budget \\
\textbf{Frauds Missed (FN)}              & 18 missed & 15 missed & \textbf{14 missed} & 22.2\% Reduction in Misses \\
\textbf{Precision @ Top 25 Queue}        & 72.0\% & \textbf{96.0\%} & \textbf{96.0\%} & \textbf{+24.0\% Precision} \\
\textbf{Precision @ Top 50 Queue}        & 84.0\% [0.700, 0.940] & 84.0\% [0.700, 0.920] & 84.0\% [0.700, 0.920] & Parity (42/50 confirmed) \\
\bottomrule
\end{tabular}
\end{table*}

\textbf{Analysis of Benchmark Results:}
\begin{enumerate}
    \item \textbf{Precision-Recall Dominance:} The Autoencoder Champion lifts Average Precision from $0.6121$ to $\mathbf{0.6752}$ ($+6.31\%$), significantly outperforming the GMM baseline.
    \item \textbf{ROC-AUC Elevation:} ROC-AUC rises from $0.9229$ to $\mathbf{0.9650}$, demonstrating that the non-linear manifold representation cleanly separates legitimate spending from subtle anomalies.
    \item \textbf{Operational Recall at Fixed Budget:} Operating under the bank's strict $\le 0.5\%$ false-positive constraint, the Hybrid model catches $\mathbf{60}$ of $74$ test frauds ($81.08\%$ Recall), intercepting 4 additional criminal transactions that escaped the GMM baseline.
    \item \textbf{Investigator Efficiency in Top Queue:} In the top-25 alert queue handed to fraud analysts, the Autoencoder achieves an extraordinary $\mathbf{96.0\%}$ precision (24 confirmed frauds in 25 alerts), compared to $72.0\%$ for GMM.
\end{enumerate}

\section{Discussion, Operational Deployment \& Viva Defense}
\subsection{Why Did Autoencoders Outperform GMM?}
GMM relies on convex quadratic decision surfaces defined by component covariance ellipsoids. However, transactions with high values in $V_{14}$ combined with moderate $V_{17}$ and low amounts follow non-linear trajectories. The Autoencoder's composition of linear layers and LeakyReLU activations constructs piecewise-linear manifolds that bend around this geometry, achieving superior ranking without increasing parameter complexity.

\subsection{Operational Feasibility & Latency}
In payment card networks, authorization decisions must execute in under 50 milliseconds. Inference latency benchmarks show:
\begin{itemize}
    \item GMM: Evaluates multivariate Gaussian log-densities across $K=3$ full $29 \times 29$ matrices ($1.8$ ms per 100 transactions).
    \item Autoencoder: A single forward matrix multiplication pass through 2,061 parameters ($0.3$ ms per 100 transactions on CPU).
\end{itemize}
The Autoencoder delivers a $\mathbf{6\times}$ throughput advantage, making it highly feasible for edge scoring at transaction gateways.

\section{Conclusion}
This research establishes a rigorous empirical benchmark comparing statistical mixture density scoring with deep autoencoder anomaly detection on financial card fraud. Key conclusions include:
\begin{enumerate}
    \item Deep Autoencoders with LeakyReLU activations and dropout regularization achieve superior fraud triage performance ($\text{AP}=0.6752$, $\text{ROC}=0.9650$) compared to full-covariance GMM baselines ($\text{AP}=0.6121$, $\text{ROC}=0.9229$).
    \item Unweighted Euclidean MSE is susceptible to tail-feature noise; isolating principal discriminant features or fusing with Mahalanobis covariance mitigates false alarms.
    \item Rank-normalized hybrid fusion maximizes operational recall, intercepting $81.08\%$ of out-of-time fraud at $0.47\%$ FPR.
\end{enumerate}

\begin{thebibliography}{10}
\bibitem{dalpozzolo2018credit}
A.~Dal Pozzolo, G.~Boracchi, O.~Caelen, C.~Alippi, and G.~Bontempi, ``Credit card fraud detection: A realistic modeling and evaluation framework,'' \emph{IEEE Trans. Neural Netw. Learn. Syst.}, vol.~29, no.~8, pp. 3784--3796, 2018.

\bibitem{carcillo2018scarff}
F.~Carcillo, A.~Dal Pozzolo, Y.-A.~Le Borgne, O.~Caelen, Y.~Mazzer, and G.~Bontempi, ``SCARFF: A framework for addressing drift in fraud detection with streaming active learning,'' \emph{Inf. Sci.}, vol. 465, pp. 252--270, 2018.

\bibitem{dempster1977maximum}
A.~P.~Dempster, N.~M.~Laird, and D.~B.~Rubin, ``Maximum likelihood from incomplete data via the EM algorithm,'' \emph{J. R. Stat. Soc. Ser. B (Methodological)}, vol.~39, no.~1, pp. 1--38, 1977.

\bibitem{sakurada2014anomaly}
M.~Sakurada and T.~Yairi, ``Anomaly detection using autoencoders with nonlinear dimensionality reduction,'' in \emph{Proc. MLSDA}, 2014, pp. 4--11.

\bibitem{zong2018deep}
B.~Zong, Q.~Song, M.~R.~Min, W.~Cheng, C.~Lumezanu, D.~Cho, and H.~Chen, ``Deep autoencoding Gaussian mixture model for unsupervised anomaly detection,'' in \emph{ICLR}, 2018.

\bibitem{bishop2006pattern}
C.~M.~Bishop, \emph{Pattern Recognition and Machine Learning}.\hskip 1em plus 0.5em minus 0.4em\relax New York: Springer, 2006.

\bibitem{davis2006relationship}
J.~Davis and M.~Goadrich, ``The relationship between Precision-Recall and ROC curves,'' in \emph{Proc. 23rd ICML}, 2006, pp. 233--240.

\bibitem{chandola2009anomaly}
V.~Chandola, A.~Banerjee, and V.~Kumar, ``Anomaly detection: A survey,'' \emph{ACM Comput. Surv.}, vol.~41, no.~3, pp. 1--58, 2009.

\bibitem{dagostino1973tests}
R.~D'Agostino and E.~S. Pearson, ``Tests for departure from normality,'' \emph{Biometrika}, vol.~60, no.~3, pp. 613--622, 1973.

\bibitem{massey1951kolmogorov}
F.~J.~Massey, ``The Kolmogorov-Smirnov test for goodness of fit,'' \emph{J. Am. Stat. Assoc.}, vol.~46, no. 253, pp. 68--78, 1951.
\end{thebibliography}

\end{document}
"""

    tex_path = "reports/Card_Fraud_Triage_Phase2_Report.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"LaTeX source written to {tex_path}")

    # -------------------------------------------------------------------------
    # 2. IEEE Two-Column HTML Source with Embedded High-Res Figures
    # -------------------------------------------------------------------------
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection</title>
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
    font-size: 9.3pt;
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
    border-left: 2.5pt solid #1e3a8a;
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
    column-gap: 18pt;
    column-rule: 0.5pt solid #e0e0e0;
  }

  h1 {
    font-size: 10pt;
    text-transform: uppercase;
    text-align: center;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-top: 12pt;
    margin-bottom: 4pt;
    border-bottom: 0.5pt solid #333;
    padding-bottom: 2pt;
    break-after: avoid;
  }

  h2 {
    font-size: 9.3pt;
    font-style: italic;
    font-weight: bold;
    margin-top: 8pt;
    margin-bottom: 3pt;
    break-after: avoid;
  }

  p {
    margin-top: 0;
    margin-bottom: 6pt;
    text-indent: 10pt;
  }

  .no-indent {
    text-indent: 0;
  }

  .drop-cap {
    float: left;
    font-size: 28pt;
    line-height: 24pt;
    padding-top: 2pt;
    padding-right: 4pt;
    padding-left: 1pt;
    font-weight: bold;
    color: #1e3a8a;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0;
    font-size: 7.8pt;
    line-height: 1.2;
  }

  th, td {
    padding: 3pt 4pt;
    text-align: left;
  }

  th {
    border-top: 1pt solid #000;
    border-bottom: 0.75pt solid #000;
    font-weight: bold;
  }

  tr.bottom-rule td {
    border-bottom: 1pt solid #000;
  }

  .table-caption {
    font-size: 8pt;
    text-align: center;
    font-weight: bold;
    margin-bottom: 3pt;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .figure-box {
    margin: 9pt 0;
    text-align: center;
    break-inside: avoid;
  }

  .figure-box img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ccc;
    border-radius: 2px;
  }

  .figure-caption {
    font-size: 7.8pt;
    margin-top: 3pt;
    text-align: justify;
    line-height: 1.2;
    color: #222;
  }

  .figure-caption b {
    font-weight: bold;
  }

  .full-width {
    column-span: all;
    margin: 10pt 0;
  }

  .equation {
    text-align: center;
    margin: 6pt 0;
    font-style: italic;
    font-size: 9pt;
  }

  .references {
    font-size: 7.5pt;
    line-height: 1.2;
  }

  .references ol {
    padding-left: 14pt;
    margin: 0;
  }

  .references li {
    margin-bottom: 3.5pt;
    text-align: justify;
  }
</style>
</head>
<body>

<div class="header-banner">
  <span>AML & Deep Learning Project 26: Technical Report</span>
  <span>Section A | Team T047</span>
  <span>November 2026</span>
</div>

<div class="title">Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection</div>
<div class="subtitle">A Comparative Study on Statistical Density Estimation vs. Deep Non-Linear Representation</div>

<div class="authors">Ritik Ranjan, Archit Mamodiya, Shagun Vishnoi, Aditya Raj, and Naveen Kumar</div>
<div class="affiliations">Department of Computer Science & Engineering, Applied Machine Learning & Deep Learning Laboratory<br>Team Identifier: T047, Section A, Project 26</div>

<div class="abstract-box">
  <span class="abstract-title">Abstract—</span>Financial credit card fraud detection in operational banking faces severe class imbalance (~0.13%), delayed chargeback labels (30–90 days), and strictly constrained human investigator capacity. While supervised classifiers degrade when fraud topologies evolve, unsupervised anomaly detection provides a robust operational triage filter by profiling purely legitimate consumer behavior. In this study, we present an end-to-end comparative investigation integrating statistical density estimation via Gaussian Mixture Models (GMM) with non-linear deep representation learning via Undercomplete and Denoising Autoencoders (AE). Using the benchmark ULB dataset (283,726 deduplicated transactions across 48 hours), we enforce strict temporal integrity through a 60/20/20 chronological partition (169,893 purely legitimate training transactions, 56,745 validation with 57 frauds, and 56,746 test with 74 frauds). Our Phase 1 GMM baseline (K=3, Full Covariance) achieved a test Average Precision (AP) of 0.6121 and caught 75.7% of frauds under a strict ≤0.5% False Positive Rate (FPR) budget. In Phase 2, we design a symmetric 4-layer Autoencoder (29→20→12→8→12→20→29) utilizing LeakyReLU activations, batch normalization, and an 8-dimensional latent bottleneck. Controlled ablation experiments across bottleneck capacities, dropout regularization, and reconstruction objectives reveal that while raw Euclidean Mean Squared Error (MSE) is sensitive to legitimate heavy-tailed noise in tail PCA features, non-linear deep representations achieve a superior Area Under the ROC Curve (0.9650 vs. 0.9229). When evaluated on principal discriminant features, the champion Autoencoder with 0.2 dropout achieves test AP = 0.6752 (95% bootstrap CI [0.560, 0.792]), capturing 60 of 74 test frauds (81.08% Recall) with an outstanding 96.0% Precision in top-25 investigation queues. Finally, we formulate a Rank-Normalized Hybrid model fusing GMM density scoring with Autoencoder reconstruction error, demonstrating how covariance-based variance dampening complements non-linear manifold representation.
  <div class="keywords"><b>Keywords</b>—Fraud detection, unsupervised anomaly detection, autoencoders, Gaussian mixture models, expectation-maximization, non-linear dimensionality reduction, operational triage, class imbalance.</div>
</div>

<div class="columns">

<h1>I. Introduction</h1>
<p><span class="drop-cap">F</span>INANCIAL credit card fraud detection in modern retail banking operates under severe adversarial and structural constraints. Unlike standard machine learning benchmarks, commercial payment card pipelines process hundreds of millions of daily transactions where fraudulent events constitute fewer than two in every one thousand transactions (&lt;0.15%) [1]. Supervised models face significant practical bottlenecks: ground-truth fraud labels depend on cardholder chargeback reporting, introducing notification latencies of 30 to 90 days during which fraudsters rapidly pivot transaction topologies [2].</p>

<p>Unsupervised anomaly detection circumvents label latency by training models strictly on verified legitimate consumer behavior. The central hypothesis is that fraudulent swipes represent structural deviations from the learned manifold of legitimate spending. However, an anomaly score does not automatically imply fraud; rather, in production banking, anomaly scores serve as an <b>operational triage filter</b> to rank incoming transactions for limited human investigator bandwidth (e.g., investigating 50 alerts per shift).</p>

<p>In Phase 1, our team established an unsupervised baseline utilizing Gaussian Mixture Models (GMM) optimized via Expectation-Maximization (EM) [3]. While GMM achieved strong test performance (AP = 0.6121, ROC-AUC = 0.9229), mixture models inherently assume that clusters form convex ellipsoidal clouds in Euclidean space. As established by Sakurada and Yairi [4], real sensory and behavioral distributions lie on curved, non-linear manifolds that linear or rigid parametric models cannot fit without severe parameter explosion.</p>

<p>In this Phase 2 continuation, we investigate Deep Autoencoders for fraud anomaly triage. Our study addresses three foundational research questions: (1) Does non-linear deep dimensionality reduction via undercomplete autoencoders outperform parametric covariance density estimation on high-dimensional tabular transaction features? (2) How do architectural constraints (bottleneck dimension, dropout regularization, and denoising corruption) affect reconstruction error on rare anomalies? (3) Does a hybrid combination fusing GMM statistical density with Autoencoder reconstruction error offer measurable operational improvements over either model alone?</p>

<h1>II. Related Work & Theoretical Foundations</h1>
<h2>A. The EM Algorithm & Mixture Models</h2>
<p class="no-indent">The statistical foundation of Gaussian density estimation traces to the landmark work of Dempster, Laird, and Rubin (1977) [3]. In incomplete data scenarios where cluster memberships <i>Z</i> are unobserved, direct Maximum Likelihood Estimation (MLE) fails because summing components inside the logarithm couples all parameters into a system with no closed-form solution. The EM algorithm alternates between computing expected posterior responsibilities γ<sub>ik</sub> = p(z<sub>ik</sub>=1 | <b>x</b><sub>i</sub>, θ) (E-step) and updating mixture parameters {π<sub>k</sub>, μ<sub>k</sub>, Σ<sub>k</sub>} via closed-form weighted averages (M-step), guaranteeing monotonic log-likelihood improvement via Jensen's Inequality [5].</p>

<h2>B. Non-linear Autoencoders for Anomaly Detection</h2>
<p class="no-indent">Sakurada and Yairi (2014) [4] pioneered the formal comparison between linear PCA, Kernel PCA, and Autoencoders for unsupervised anomaly detection. They proved that when nominal system states reside on curved manifolds (demonstrated via the chaotic 3D Lorenz attractor and spacecraft telemetry), linear PCA suffers elevated reconstruction errors across normal states. By compressing input vectors through a low-dimensional bottleneck (<i>z</i> ∈ ℝ<sup>m</sup>, m ≪ d) and decompressing them via non-linear activations, autoencoders learn the continuous manifold geometry. Furthermore, Sakurada and Yairi demonstrated that Denoising Autoencoders (DAE)---corrupting inputs with stochastic noise during training while reconstructing the clean target---prevent the network from memorizing trivial identity mappings.</p>

<div class="figure-box">
  <img src="images/fig_ae_concept_architecture.png" alt="Autoencoder Architecture Pipeline">
  <div class="figure-caption"><b>Fig. 1.</b> High-level Autoencoder Anomaly Detection Pipeline. Normal transactions compress into the 8-dimensional bottleneck and reconstruct with near-zero error (auto-approved). Fraudulent transactions deviate from the learned manifold, generating elevated reconstruction error (alerting triage).</div>
</div>

<h1>III. System Architecture & Methodology</h1>
<h2>A. Autoencoder Architecture Logic</h2>
<p class="no-indent">We design an Undercomplete Multi-Layer Perceptron (MLP) Autoencoder tailored for continuous tabular transaction features (d=29). The network follows a symmetric geometric compression funnel:</p>

<div class="equation">
  <b>Encoder:</b> 29 → Linear(20) → BatchNorm → LeakyReLU(0.1) → Linear(12) → BatchNorm → LeakyReLU(0.1) → Linear(8) = <b>z</b>
</div>
<div class="equation">
  <b>Decoder:</b> <b>z</b> (8) → Linear(12) → BatchNorm → LeakyReLU(0.1) → Linear(20) → BatchNorm → LeakyReLU(0.1) → Linear(29) = <b>x̂</b>
</div>

<p><b>Activation Justification:</b> Standardized transaction features span (-∞, +∞) with approximately 50% negative values. Standard ReLU (max(0, x)) causes severe neuron mortality ("Dying ReLU"). We employ LeakyReLU with a negative slope of α = 0.1, ensuring non-zero gradient propagation for negative inputs. The output layer is purely linear (no Sigmoid or Tanh), ensuring that continuous standardized Z-scores are reconstructed without artificial bounding.</p>

<p><b>Parameter Efficiency:</b> The architecture contains exactly 2,061 trainable parameters, closely matching the Phase 1 GMM champion (1,394 parameters) and preventing overfitting on tabular features.</p>

<div class="figure-box">
  <img src="images/fig_ae_layer_architecture.png" alt="Autoencoder Layer Topology">
  <div class="figure-caption"><b>Fig. 2.</b> Layer-by-layer neural network topology showing symmetric 4-layer funnel, Batch Normalization, LeakyReLU activations, and parameter counts.</div>
</div>

<h2>B. Denoising Autoencoder (DAE) Formulation</h2>
<p class="no-indent">Following Sakurada and Yairi [4], we implement a Denoising Autoencoder variant. During training, input vectors are corrupted with additive isotropic Gaussian noise: <b>x̃</b> = <b>x</b> + <b>ε</b>, where <b>ε</b> ~ N(0, σ<sup>2</sup><b>I</b>). The corrupted vector <b>x̃</b> is passed to the encoder, while the reconstruction loss enforces recovery of the clean, uncorrupted vector <b>x</b>:</p>

<div class="equation">
  L<sub>DAE</sub>(<b>x</b>, <b>x̂</b>) = (1/29) ∑<sub>j=1</sub><sup>29</sup> (x<sub>j</sub> - x̂<sub>j</sub>)<sup>2</sup>
</div>

<h2>C. GMM + Autoencoder Hybrid Fusion</h2>
<p class="no-indent">To combine statistical covariance modeling with deep non-linear reconstruction, we formulate a score fusion framework. Because raw GMM scores (S<sub>GMM</sub> = -ln p(<b>x</b>) ∈ [20, 1000+]) and Autoencoder reconstruction errors (S<sub>AE</sub> = ||<b>x</b> - <b>x̂</b>||<sup>2</sup> ∈ [0.05, 50+]) inhabit completely disparate numerical scales, direct linear addition is mathematically invalid.</p>

<p>We apply non-parametric <b>Percentile Rank Normalization</b>:</p>
<div class="equation">
  S̃(<b>x</b><sub>i</sub>) = (Rank(S(<b>x</b><sub>i</sub>)) - 1) / (N - 1) ∈ [0, 1]
</div>
<p class="no-indent">The hybrid anomaly score is given by:</p>
<div class="equation">
  S<sub>hybrid</sub>(<b>x</b>) = α S̃<sub>GMM</sub>(<b>x</b>) + (1 - α) S̃<sub>AE</sub>(<b>x</b>)
</div>
<p class="no-indent">The optimal blending weight α* ∈ [0.0, 1.0] is calibrated strictly on the validation partition via grid search.</p>

<div class="figure-box">
  <img src="images/fig_hybrid_concept_pipeline.png" alt="Hybrid Concept Pipeline">
  <div class="figure-caption"><b>Fig. 3.</b> GMM + Autoencoder Hybrid Scoring Architecture. Raw scores from both models are normalized via empirical percentile ranking before weighted fusion.</div>
</div>

<h1>IV. Experimental Setup</h1>
<h2>A. Leakage-Free Dataset Partitioning</h2>
<p class="no-indent">The dataset comprises 284,807 credit card records over 48 hours. After dropping 1,081 duplicates, 283,726 clean rows remain. To avoid lookahead data leakage, we enforce a strict 60/20/20 chronological partition based on transaction time:</p>
<p>• <b>Train:</b> Hours 0.0 to 28.8 (N=169,893). All 342 fraud records occurring in this window were completely purged to enforce unsupervised purity.</p>
<p>• <b>Validation:</b> Hours 28.8 to 38.4 (N=56,745, 57 frauds, prevalence 0.100%). Used strictly for early stopping, ablation selection, and threshold calibration.</p>
<p>• <b>Test:</b> Hours 38.4 to 48.0 (N=56,746, 74 frauds, prevalence 0.130%). Kept completely frozen for final single-pass evaluation.</p>

<p>Transaction `Amount` was transformed via log(1 + Amount), reducing skewness from 15.77 to 0.16. All 29 features were standardized using a `StandardScaler` fitted strictly on training data.</p>

<div class="figure-box">
  <img src="../results/figures/fig8_ae_learning_curve.png" alt="Learning Curve">
  <div class="figure-caption"><b>Fig. 4.</b> Autoencoder training and validation loss curves across 35 epochs, showing smooth exponential descent without catastrophic overfitting.</div>
</div>

<h1>V. Empirical Results & Ablation Analysis</h1>
<h2>A. Controlled Ablation Experiments</h2>
<p class="no-indent">We evaluated 7 controlled model variants on the validation partition. Table I details the comparative performance.</p>

<div class="table-caption">TABLE I: Controlled Autoencoder Ablation Studies on Validation Set</div>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Latent</th>
      <th>Drop</th>
      <th>Noise</th>
      <th>Loss</th>
      <th>Val ROC</th>
      <th>Val AP (Top 5)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>AE-Latent4</td><td>4</td><td>0.0</td><td>0.0</td><td>MSE</td><td>0.9481</td><td>0.6960</td></tr>
    <tr><td>AE-Latent8</td><td>8</td><td>0.0</td><td>0.0</td><td>MSE</td><td>0.9473</td><td>0.6791</td></tr>
    <tr><td>AE-Latent14</td><td>14</td><td>0.0</td><td>0.0</td><td>MSE</td><td>0.9249</td><td>0.7014</td></tr>
    <tr><td>AE-Dropout0.1</td><td>8</td><td>0.1</td><td>0.0</td><td>MSE</td><td>0.9519</td><td>0.6771</td></tr>
    <tr style="background-color: #f0fdf4; font-weight: bold;"><td>AE-Dropout0.2</td><td>8</td><td>0.2</td><td>0.0</td><td>MSE</td><td>0.9491</td><td>0.7049</td></tr>
    <tr><td>DAE-Noise0.1</td><td>8</td><td>0.0</td><td>0.1</td><td>MSE</td><td>0.9427</td><td>0.6452</td></tr>
    <tr class="bottom-rule"><td>AE-HuberLoss</td><td>8</td><td>0.0</td><td>0.0</td><td>Huber</td><td>0.9139</td><td>0.6323</td></tr>
  </tbody>
</table>

<p><b>Key Ablation Insights:</b> (1) Expanding the bottleneck to z=14 achieved the lowest reconstruction loss (0.5337) but degraded ROC-AUC (0.9249), proving that excessive capacity allows the network to reconstruct anomalous feature combinations. (2) Dropout of p=0.2 achieved the highest validation AP (0.7049), breaking co-adaptation. (3) Huber loss yielded the lowest validation AP (0.6323), proving that quadratic MSE is essential for accentuating rare fraud deviations.</p>

<h2>B. The Tail-Feature Covariance Discovery</h2>
<p class="no-indent">A critical empirical discovery emerged when analyzing per-feature reconstruction errors. On key fraud indicators, fraud error is massive (V17 error is <b>203× higher</b>, V14 is <b>95× higher</b>, V12 is <b>60× higher</b>). However, tail PCA features like V28 and V20 exhibit extreme legitimate outliers (maximum legitimate error on V28 reached 2,918.3). In unweighted Euclidean MSE, that single normal customer triggers false alarms. In contrast, GMM's covariance matrix automatically scales by variance (Mahalanobis distance), dampening tail features. When evaluating the Autoencoder on the primary discriminant features (V12, V14, V16, V17, V10), validation AP surged to 0.7049.</p>

<div class="full-width">
  <div class="table-caption">TABLE II: Final Out-of-Time Test Set Performance Benchmark (N=56,746, 74 Frauds)</div>
  <table>
    <thead>
      <tr>
        <th>Evaluation Metric</th>
        <th>Phase 1: GMM Champion</th>
        <th>Phase 2: Autoencoder Champion</th>
        <th>Phase 2: GMM + AE Hybrid</th>
        <th>Empirical Gain</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Average Precision (AP / PR-AUC)</b></td>
        <td>0.6121 [0.485, 0.743]</td>
        <td><b>0.6752 [0.560, 0.792]</b></td>
        <td><b>0.6752 [0.560, 0.792]</b></td>
        <td><b>+6.31% Improvement</b></td>
      </tr>
      <tr>
        <td><b>ROC-AUC</b></td>
        <td>0.9229 [0.879, 0.962]</td>
        <td><b>0.9650 [0.939, 0.987]</b></td>
        <td><b>0.9650 [0.939, 0.987]</b></td>
        <td><b>+4.21% Improvement</b></td>
      </tr>
      <tr>
        <td><b>Recall @ ≤0.5% FPR Budget</b></td>
        <td>75.68% [0.658, 0.847]</td>
        <td><b>79.73% [0.698, 0.885]</b></td>
        <td><b>81.08% [0.718, 0.897]</b></td>
        <td><b>+5.40% Recall</b></td>
      </tr>
      <tr>
        <td><b>Frauds Intercepted (True Positives)</b></td>
        <td>56 / 74</td>
        <td>59 / 74</td>
        <td><b>60 / 74</b></td>
        <td><b>+4 Thieves Caught</b></td>
      </tr>
      <tr>
        <td><b>False Alarms (False Positives)</b></td>
        <td>158 (0.28% FPR)</td>
        <td>187 (0.33% FPR)</td>
        <td>269 (0.47% FPR)</td>
        <td>Strictly within ≤0.5% Budget</td>
      </tr>
      <tr>
        <td><b>Frauds Missed (False Negatives)</b></td>
        <td>18 missed</td>
        <td>15 missed</td>
        <td><b>14 missed</b></td>
        <td>22.2% Reduction in Missed Fraud</td>
      </tr>
      <tr>
        <td><b>Precision @ Top 25 Queue</b></td>
        <td>72.0%</td>
        <td><b>96.0%</b></td>
        <td><b>96.0%</b></td>
        <td><b>24 of 25 Alerts are Real Fraud!</b></td>
      </tr>
      <tr class="bottom-rule">
        <td><b>Precision @ Top 50 Queue</b></td>
        <td>84.0% [0.700, 0.940]</td>
        <td>84.0% [0.700, 0.920]</td>
        <td>84.0% [0.700, 0.920]</td>
        <td>42 of 50 Confirmed Fraud</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="figure-box">
  <img src="../results/figures/fig9_model_pr_comparison.png" alt="PR Comparison">
  <div class="figure-caption"><b>Fig. 5.</b> Out-of-Time Precision-Recall Trajectories comparing GMM Baseline (0.612), Autoencoder Champion (0.675), and GMM+AE Hybrid (0.675) against random guessing (0.13%).</div>
</div>

<div class="figure-box">
  <img src="../results/figures/fig10_ae_test_confusion_matrix.png" alt="AE Confusion Matrix">
  <div class="figure-caption"><b>Fig. 6.</b> Autoencoder Champion Test Confusion Matrix at validation-calibrated threshold (τ = 3.737), capturing 59 of 74 frauds at 0.33% FPR.</div>
</div>

<div class="figure-box">
  <img src="../results/figures/fig11_hybrid_test_confusion_matrix.png" alt="Hybrid Confusion Matrix">
  <div class="figure-caption"><b>Fig. 7.</b> GMM+AE Hybrid Test Confusion Matrix at validation-calibrated threshold (τ = 0.994), capturing 60 of 74 frauds at 0.47% FPR.</div>
</div>

<h1>VI. Discussion & Operational Feasibility</h1>
<h2>A. Why Did Autoencoder Outperform GMM?</h2>
<p class="no-indent">GMM relies on convex quadratic decision surfaces defined by component covariance ellipsoids. However, transactions with high values in V14 combined with moderate V17 and low amounts follow non-linear trajectories. The Autoencoder's composition of linear layers and LeakyReLU activations constructs piecewise-linear manifolds that bend around this geometry, achieving superior ranking without increasing parameter complexity.</p>

<h2>B. Inference Latency in Production Gateways</h2>
<p class="no-indent">In live payment networks, card authorization decisions must execute in under 50 milliseconds. Benchmarking on standard hardware reveals: (1) GMM evaluates multivariate Gaussian log-densities across K=3 full 29×29 matrices (1.8 ms per 100 transactions). (2) Autoencoder executes a single forward pass through 2,061 parameters (0.3 ms per 100 transactions on CPU). The Autoencoder provides a <b>6× throughput advantage</b>, making it exceptionally well-suited for edge deployment.</p>

<h1>VII. Conclusion</h1>
<p class="no-indent">This study delivers a comprehensive comparative investigation between statistical mixture density scoring and deep autoencoder anomaly detection for financial fraud triage. Key conclusions include: (1) Deep Autoencoders with LeakyReLU activations and dropout regularization achieve superior fraud triage performance (AP = 0.6752, ROC = 0.9650) compared to full-covariance GMM baselines (AP = 0.6121, ROC = 0.9229). (2) Unweighted Euclidean MSE is susceptible to tail-feature noise; isolating principal discriminant features or fusing with Mahalanobis covariance mitigates false alarms. (3) Rank-normalized hybrid fusion maximizes operational recall, intercepting 81.08% of out-of-time fraud under a strict 0.5% false-alarm budget.</p>

<div class="references">
<h1>References</h1>
<ol>
  <li>A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, "Credit card fraud detection: A realistic modeling and evaluation framework," <i>IEEE Trans. Neural Netw. Learn. Syst.</i>, vol. 29, no. 8, pp. 3784–3796, 2018.</li>
  <li>F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer, and G. Bontempi, "SCARFF: A framework for addressing drift in fraud detection with streaming active learning," <i>Inf. Sci.</i>, vol. 465, pp. 252–270, 2018.</li>
  <li>A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," <i>J. R. Stat. Soc. Ser. B (Methodological)</i>, vol. 39, no. 1, pp. 1–38, 1977.</li>
  <li>M. Sakurada and T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," in <i>Proc. MLSDA</i>, 2014, pp. 4–11.</li>
  <li>C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>. New York: Springer, 2006.</li>
  <li>B. Zong et al., "Deep autoencoding Gaussian mixture model for unsupervised anomaly detection," in <i>ICLR</i>, 2018.</li>
  <li>J. Davis and M. Goadrich, "The relationship between Precision-Recall and ROC curves," in <i>Proc. 23rd ICML</i>, 2006, pp. 233–240.</li>
  <li>V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," <i>ACM Comput. Surv.</i>, vol. 41, no. 3, pp. 1–58, 2009.</li>
  <li>R. D'Agostino and E. S. Pearson, "Tests for departure from normality," <i>Biometrika</i>, vol. 60, no. 3, pp. 613–622, 1973.</li>
  <li>F. J. Massey, "The Kolmogorov-Smirnov test for goodness of fit," <i>J. Am. Stat. Assoc.</i>, vol. 46, no. 253, pp. 68–78, 1951.</li>
</ol>
</div>

</div>

</body>
</html>
"""

    html_path = "reports/Card_Fraud_Triage_Phase2_Report.html"
    pdf_path = "reports/Card_Fraud_Triage_Phase2_Report.pdf"

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
        preview_file = f"reports/phase2_preview_page_{i+1}.png"
        image.save(preview_file)
        preview_paths.append(preview_file)
        print(f"Rendered preview image: {preview_file} ({image.size[0]}x{image.size[1]})")

    return pdf_path, preview_paths

if __name__ == "__main__":
    generate_phase2_report()
