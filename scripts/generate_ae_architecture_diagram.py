"""
Generate a professional, high-resolution architecture & concept diagram
for the Autoencoder anomaly detection pipeline.
Saved to results/figures/fig_ae_concept_architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_autoencoder_diagram():
    os.makedirs('results/figures', exist_ok=True)
    
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    
    # Background color
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Title Banner
    ax.text(6.5, 8.0, "How an Autoencoder Catches Credit Card Fraud", 
            fontsize=18, fontweight='bold', ha='center', va='center', color='#111827', fontfamily='sans-serif')
    ax.text(6.5, 7.6, "Normal transactions compress & rebuild cleanly. Fraud cannot be reconstructed, triggering the alarm.", 
            fontsize=11, ha='center', va='center', color='#4b5563', style='italic')

    # Styles
    box_blue = dict(boxstyle="round,pad=0.5", fc="#eff6ff", ec="#3b82f6", lw=2)
    box_green = dict(boxstyle="round,pad=0.5", fc="#f0fdf4", ec="#22c55e", lw=2)
    box_red = dict(boxstyle="round,pad=0.5", fc="#fef2f2", ec="#ef4444", lw=2)
    box_purple = dict(boxstyle="round,pad=0.5", fc="#faf5ff", ec="#a855f7", lw=2.5)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#fffbeb", ec="#f59e0b", lw=2)
    box_gray = dict(boxstyle="round,pad=0.4", fc="#f3f4f6", ec="#9ca3af", lw=1.5)

    # -------------------------------------------------------------
    # SCENARIO 1: NORMAL TRANSACTION (HONEST CUSTOMER) - Y = 5.2
    # -------------------------------------------------------------
    y_norm = 5.3
    
    # Track label
    ax.text(0.5, y_norm + 1.2, "SCENARIO A: Normal Cardholder Spending", 
            fontsize=12, fontweight='bold', color="#166534", ha='left')
    
    # Background panel for Normal
    panel_norm = patches.FancyBboxPatch((0.4, y_norm - 1.1), 12.2, 2.1,
                                        boxstyle="round,pad=0.1", fc="#f8fafc", ec="#cbd5e1", lw=1.2, linestyle="--")
    ax.add_patch(panel_norm)
    
    # 1. Normal Input Box
    ax.text(1.5, y_norm, "Normal Transaction\n$\\mathbf{x}$ (29 Features)\n\n• Groceries / Gas\n• Routine Amount\n• Familiar Pattern", 
            ha='center', va='center', fontsize=9.5, bbox=box_blue, fontweight='medium')
    
    # Arrow to Encoder
    ax.annotate('', xy=(3.0, y_norm), xytext=(2.4, y_norm),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#3b82f6"))
    
    # 2. Encoder Funnel
    ax.text(3.7, y_norm, "ENCODER\n\n$29 \\to 20$\n$\\to 12$", 
            ha='center', va='center', fontsize=9.5, bbox=box_gray, fontweight='bold', color="#1f2937")
    
    # Arrow to Bottleneck
    ax.annotate('', xy=(5.0, y_norm), xytext=(4.4, y_norm),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#3b82f6"))
    
    # 3. Latent Bottleneck (The 8 numbers)
    ax.text(5.8, y_norm, "Bottleneck $\\mathbf{z}$\n(8 Numbers)\n\nSummary Code\n(Normal Habit)", 
            ha='center', va='center', fontsize=9.5, bbox=box_purple, fontweight='bold', color="#6b21a8")
    
    # Arrow to Decoder
    ax.annotate('', xy=(7.2, y_norm), xytext=(6.6, y_norm),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#a855f7"))
    
    # 4. Decoder Horn
    ax.text(7.9, y_norm, "DECODER\n\n$8 \\to 12$\n$\\to 20$", 
            ha='center', va='center', fontsize=9.5, bbox=box_gray, fontweight='bold', color="#1f2937")
    
    # Arrow to Reconstruction
    ax.annotate('', xy=(9.2, y_norm), xytext=(8.6, y_norm),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#22c55e"))
    
    # 5. Reconstruction Box
    ax.text(9.9, y_norm, "Reconstructed $\\mathbf{\\hat{x}}$\n(29 Features)\n\n• Matches Original!\n• Rebuild is clean", 
            ha='center', va='center', fontsize=9.5, bbox=box_green, fontweight='medium')

    # Arrow to Verdict
    ax.annotate('', xy=(11.1, y_norm), xytext=(10.7, y_norm),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#22c55e"))

    # 6. Verdict Badge
    ax.text(11.8, y_norm, "Error ~ 0\n\nAPPROVED\n(Legitimate)", 
            ha='center', va='center', fontsize=10, fontweight='bold', color="#15803d",
            bbox=dict(boxstyle="round,pad=0.5", fc="#dcfce7", ec="#16a34a", lw=2.5))


    # -------------------------------------------------------------
    # SCENARIO 2: FRAUDULENT TRANSACTION (THE THIEF) - Y = 2.4
    # -------------------------------------------------------------
    y_fraud = 2.5
    
    # Track label
    ax.text(0.5, y_fraud + 1.2, "SCENARIO B: Thief / Fraudulent Transaction", 
            fontsize=12, fontweight='bold', color="#991b1b", ha='left')
    
    # Background panel for Fraud
    panel_fraud = patches.FancyBboxPatch((0.4, y_fraud - 1.1), 12.2, 2.1,
                                         boxstyle="round,pad=0.1", fc="#fff5f5", ec="#fca5a5", lw=1.2, linestyle="--")
    ax.add_patch(panel_fraud)
    
    # 1. Fraud Input Box
    ax.text(1.5, y_fraud, "Fraud Transaction\n$\\mathbf{x}$ (29 Features)\n\n• 3 AM Spike\n• Atypical Location\n• Weird PCA Mix", 
            ha='center', va='center', fontsize=9.5, bbox=box_red, fontweight='medium')
    
    # Arrow to Encoder
    ax.annotate('', xy=(3.0, y_fraud), xytext=(2.4, y_fraud),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#ef4444"))
    
    # 2. Encoder Funnel
    ax.text(3.7, y_fraud, "ENCODER\n\n(Trained ONLY\non normal data)", 
            ha='center', va='center', fontsize=9.5, bbox=box_gray, fontweight='bold', color="#1f2937")
    
    # Arrow to Bottleneck
    ax.annotate('', xy=(5.0, y_fraud), xytext=(4.4, y_fraud),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#ef4444"))
    
    # 3. Latent Bottleneck (Fails to compress)
    ax.text(5.8, y_fraud, "Bottleneck $\\mathbf{z}$\n(8 Numbers)\n\nDistorted Code!\n(No normal slot)", 
            ha='center', va='center', fontsize=9.5, bbox=box_orange, fontweight='bold', color="#b45309")
    
    # Arrow to Decoder
    ax.annotate('', xy=(7.2, y_fraud), xytext=(6.6, y_fraud),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#f59e0b"))
    
    # 4. Decoder Horn
    ax.text(7.9, y_fraud, "DECODER\n\n(Tries to guess\nnormal shapes)", 
            ha='center', va='center', fontsize=9.5, bbox=box_gray, fontweight='bold', color="#1f2937")
    
    # Arrow to Reconstruction
    ax.annotate('', xy=(9.2, y_fraud), xytext=(8.6, y_fraud),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#ef4444"))
    
    # 5. Reconstruction Box (Failed!)
    ax.text(9.9, y_fraud, "Mismatched $\\mathbf{\\hat{x}}$\n(29 Features)\n\n• Fails to rebuild!\n• Huge differences", 
            ha='center', va='center', fontsize=9.5, bbox=box_red, fontweight='medium')

    # Arrow to Verdict
    ax.annotate('', xy=(11.1, y_fraud), xytext=(10.7, y_fraud),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#ef4444"))

    # 6. Verdict Badge
    ax.text(11.8, y_fraud, "Error is HUGE!\n\nALERT!\n(Fraud Triage)", 
            ha='center', va='center', fontsize=10, fontweight='bold', color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fee2e2", ec="#dc2626", lw=2.5))

    # -------------------------------------------------------------
    # BOTTOM BANNER: THE MATHEMATICAL LOGIC
    # -------------------------------------------------------------
    math_box = patches.FancyBboxPatch((1.5, 0.3), 10.0, 0.85,
                                      boxstyle="round,pad=0.2", fc="#f8fafc", ec="#94a3b8", lw=1.5)
    ax.add_patch(math_box)
    ax.text(6.5, 0.72, 
            r"Reconstruction Error (Anomaly Score) = $\frac{1}{29} \sum_{j=1}^{29} (x_j - \hat{x}_j)^2$" + 
            "    |    " + 
            r"Threshold $\tau$: Score $\geq \tau \rightarrow$ Alert Fraud Triage",
            ha='center', va='center', fontsize=11, color="#0f172a", fontweight='bold')

    plt.tight_layout()
    output_path = 'results/figures/fig_ae_concept_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated clean architecture infographic at: {output_path}")

if __name__ == '__main__':
    draw_autoencoder_diagram()
