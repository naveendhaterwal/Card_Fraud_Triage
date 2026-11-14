"""
Generate a clean, high-resolution conceptual infographic for the
GMM + Autoencoder Hybrid Anomaly Detection Pipeline.
Saved to:
  results/figures/fig_hybrid_concept_pipeline.png
  notebooks/images/fig_hybrid_concept_pipeline.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_hybrid_diagram():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('notebooks/images', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Title
    ax.text(6.5, 7.0, "GMM + Autoencoder Hybrid Scoring Architecture", 
            fontsize=17, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(6.5, 6.55, "Combining Statistical Covariance Density with Deep Non-Linear Manifold Reconstruction", 
            fontsize=10.5, ha='center', va='center', color='#475569', style='italic')

    # Styles
    b_input = dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#2563eb", lw=2)
    b_gmm   = dict(boxstyle="round,pad=0.5", fc="#f0f9ff", ec="#0284c7", lw=2)
    b_ae    = dict(boxstyle="round,pad=0.5", fc="#faf5ff", ec="#7c3aed", lw=2)
    b_norm  = dict(boxstyle="round,pad=0.4", fc="#fffbeb", ec="#f59e0b", lw=1.8)
    b_fuse  = dict(boxstyle="round,pad=0.6", fc="#fdf2f8", ec="#db2777", lw=2.5)
    b_alert = dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#dc2626", lw=2.5)
    b_safe  = dict(boxstyle="round,pad=0.5", fc="#f0fdf4", ec="#16a34a", lw=2.5)

    # 1. Input Box (Top Center)
    ax.text(6.5, 5.7, "Incoming Transaction x (29 Features)\nStandardized: V1..V28 + log(Amount)", 
            ha='center', va='center', fontsize=10, bbox=b_input, fontweight='bold', color="#1e3a8a")

    # Diverging Arrows from Input
    ax.annotate('', xy=(3.5, 4.6), xytext=(5.3, 5.4),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#0284c7"))
    ax.annotate('', xy=(9.5, 4.6), xytext=(7.7, 5.4),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#7c3aed"))

    # 2. Branch 1: GMM (Left)
    ax.text(3.5, 4.2, "Model 1: Gaussian Mixture (GMM)\n\n• K=3 Full Covariance Components\n• Mahalanobis Distance\n• Dampens noisy tail dimensions", 
            ha='center', va='center', fontsize=9, bbox=b_gmm, color="#0369a1")
    
    # 3. Branch 2: Autoencoder (Right)
    ax.text(9.5, 4.2, "Model 2: Deep Autoencoder (AE)\n\n• 4-Layer Non-Linear Funnel\n• Compresses to 8-dim Bottleneck\n• Captures curved fraud manifolds", 
            ha='center', va='center', fontsize=9, bbox=b_ae, color="#5b21b6")

    # Down Arrows to Raw Scores
    ax.annotate('', xy=(3.5, 3.1), xytext=(3.5, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#0284c7"))
    ax.annotate('', xy=(9.5, 3.1), xytext=(9.5, 3.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed"))

    # Raw Score Badges
    ax.text(3.5, 2.8, "Raw Score: -ln p(x)\n(Span: 20 to 1,000+)", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#0369a1")
    ax.text(9.5, 2.8, "Raw Score: ||x - x^||²\n(Span: 0.05 to 50+)", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color="#5b21b6")

    # Down Arrows to Normalization
    ax.annotate('', xy=(3.5, 2.2), xytext=(3.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#f59e0b"))
    ax.annotate('', xy=(9.5, 2.2), xytext=(9.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#f59e0b"))

    # 4. Normalization Boxes
    ax.text(3.5, 1.8, "Percentile Rank Norm\nS_gmm in [0, 1]", 
            ha='center', va='center', fontsize=8.5, bbox=b_norm, color="#b45309", fontweight='bold')
    ax.text(9.5, 1.8, "Percentile Rank Norm\nS_ae in [0, 1]", 
            ha='center', va='center', fontsize=8.5, bbox=b_norm, color="#b45309", fontweight='bold')

    # Converging Arrows to Fusion
    ax.annotate('', xy=(5.5, 1.0), xytext=(4.4, 1.6),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#db2777"))
    ax.annotate('', xy=(7.5, 1.0), xytext=(8.6, 1.6),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#db2777"))

    # 5. Fusion Center Box
    ax.text(6.5, 0.85, "HYBRID FUSION ENGINE\n\nS_hybrid = alpha * S_gmm + (1 - alpha) * S_ae\n(Optimal alpha = 0.50 calibrated on Validation)", 
            ha='center', va='center', fontsize=10, bbox=b_fuse, color="#831843", fontweight='bold')

    # Decision Arrows (Out to Sides)
    ax.annotate('', xy=(1.5, 0.85), xytext=(4.3, 0.85),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#dc2626"))
    ax.annotate('', xy=(11.5, 0.85), xytext=(8.7, 0.85),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#16a34a"))

    # Decisions
    ax.text(1.3, 0.85, "Score >= tau\n\nALERT!\n(Fraud Triage)", 
            ha='center', va='center', fontsize=9.5, bbox=b_alert, color="#991b1b", fontweight='bold')
    ax.text(11.7, 0.85, "Score < tau\n\nAPPROVE\n(Normal)", 
            ha='center', va='center', fontsize=9.5, bbox=b_safe, color="#166534", fontweight='bold')

    plt.tight_layout()
    for p in ['results/figures/fig_hybrid_concept_pipeline.png', 'notebooks/images/fig_hybrid_concept_pipeline.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated fig_hybrid_concept_pipeline.png successfully!")

if __name__ == '__main__':
    draw_hybrid_diagram()
