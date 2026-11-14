# Research Study Guide: Sakurada & Yairi (2014)

**Title:** Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction  
**Authors:** Mayu Sakurada and Takehisa Yairi (The University of Tokyo / JAXA)  
**Published In:** *Proceedings of the 2nd Workshop on Machine Learning for Sensory Data Analysis (MLSDA '14)*, ACM, pp. 4–11.  
**DOI:** [10.1145/2689746.2689747](https://doi.org/10.1145/2689746.2689747)

---

## 1. Executive Summary & Core Hypothesis

Traditional anomaly detection often relies on **linear dimensionality reduction** like Principal Component Analysis (PCA). PCA assumes that normal data points lie on a flat, linear subspace (hyperplane). 

Sakurada and Yairi demonstrated that:
1. Real-world systems generate data on **curved, non-linear manifolds**. Linear PCA cannot bend to fit these structures, leading to high reconstruction error even for normal observations, or false negatives on subtle anomalies.
2. An **Autoencoder (AE)** with non-linear activation functions acts as a **non-linear generalization of PCA**. By compressing data through a low-dimensional bottleneck layer and decompressing it, the autoencoder learns the non-linear manifold of normal behavior.
3. When trained *strictly on normal data*, the autoencoder learns to reconstruct normal patterns with near-zero error. When an anomalous pattern is fed in, the network fails to reconstruct it, producing a large **reconstruction error** that serves as a direct anomaly score:
   $$\text{Anomaly Score}(x) = \| x - \hat{x} \|^2 = \sum_{j=1}^d (x_j - \hat{x}_j)^2$$
4. **Denoising Autoencoders (DAE)**—which inject small noise into inputs during training while forcing the model to reconstruct the clean input—prevent the autoencoder from memorizing trivial identity mappings and significantly improve anomaly detection on noisy real-world data.
5. While **Kernel PCA (KPCA)** also captures non-linear manifolds, KPCA requires $O(N^3)$ training complexity and storing all support vectors, scaling poorly to large datasets. Autoencoders provide $O(1)$ constant-time forward-pass inference per sample.

---

## 2. Mathematical Formulation

### 2.1 The Autoencoder Architecture
- **Input vector:** $x \in \mathbb{R}^d$
- **Encoder:** Maps $x$ to a lower-dimensional latent representation $z \in \mathbb{R}^m$ ($m < d$):
  $$z = f_\theta(x) = \sigma(W_e x + b_e)$$
- **Decoder:** Reconstructs the original input from latent space:
  $$\hat{x} = g_{\theta'}(z) = \sigma'(W_d z + b_d)$$
- **Reconstruction Loss (MSE):**
  $$L(x, \hat{x}) = \frac{1}{d} \sum_{j=1}^d (x_j - \hat{x}_j)^2$$

### 2.2 Denoising Autoencoder (DAE) Formulation
1. Corrupt clean input $x$ with stochastic noise: $\tilde{x} \sim q_D(\tilde{x} \mid x)$ (e.g., additive Gaussian noise $\tilde{x} = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ or dropout mask).
2. Encode corrupted input: $z = f_\theta(\tilde{x})$.
3. Decode to reconstruct the **uncorrupted** original: $\hat{x} = g_{\theta'}(z)$.
4. Loss is computed against the *clean* target:
   $$L_{\text{DAE}} = \| x - \hat{x} \|^2$$

---

## 3. Experimental Methodology & Results in the Paper

Sakurada & Yairi validated their framework on two domains:

1. **Lorenz Chaotic Attractor (Synthetic 3D System):**
   - Governing equations form non-linear butterfly-wing manifolds:
     $$\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z$$
   - PCA completely failed to separate subtle trajectory anomalies from normal state variations because the manifold is severely folded and non-linear.
   - Non-linear Autoencoders captured the continuous manifold geometry, isolating trajectory anomalies with high precision.

2. **JAXA Spacecraft Telemetry Data (Real-World Sensory Data):**
   - Telemetry from Japanese satellite subsystems (temperatures, voltages, motor currents).
   - DAE achieved higher Area Under the ROC Curve (AUC) than linear PCA and Kernel PCA across multiple failure modes (power drop anomalies, sensor bias drift).

---

## 4. Direct Connection to Team T047 Fraud Triage

| Aspect in Sakurada & Yairi (2014) | Team T047 Implementation (Phase 2) |
|---|---|
| **Input Data** | Spacecraft sensory telemetry ($d$ dimensions) | Credit card transaction features ($d=29$: `V1`–`V28` + $\log(1+\text{Amount})$) |
| **Normal Training** | Nominal spacecraft flight logs | $169,893$ verified legitimate cardholder transactions (`train_legit.csv`) |
| **Anomaly Metric** | Squared Euclidean reconstruction error $\|x - \hat{x}\|^2$ | Mean Squared Error (MSE) / Mean Absolute Error (MAE) reconstruction loss |
| **Regularization** | Denoising Autoencoder (DAE) with input noise | Denoising Autoencoder + Weight Decay (L2) + Dropout in PyTorch |
| **Baseline Comparison** | Linear PCA & Kernel PCA | Phase 1 Gaussian Mixture Model (GMM) with $K=3$ Full Covariance |

---

## 5. Viva Defense Card: Why Sakurada & Yairi Matters for Our Project

- **Faculty Question:** *"Why choose an Autoencoder over PCA or GMM for credit card fraud?"*
- **Defense Answer:**  
  *"As Sakurada & Yairi (2014) proved, linear methods like PCA and ellipsoidal models like GMM assume normal data lies on hyperplanes or convex Gaussian clouds. Credit card customer behavior spans complex, non-linear manifolds (e.g., correlation between spending velocity, time-of-day, and transaction amount). An Autoencoder with non-linear activations learns this curved manifold. An anomalous fraud transaction deviates from this manifold, resulting in a significantly elevated reconstruction error."*
