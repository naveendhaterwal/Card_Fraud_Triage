"""
Generate a publication-quality neural network layer architecture diagram.
Saved to:
  results/figures/fig_ae_layer_architecture.png
  notebooks/images/fig_ae_layer_architecture.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_layer_diagram():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('notebooks/images', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    # Title
    ax.text(7.5, 5.4, "Autoencoder Layer-by-Layer Architecture (Total: 2,061 Parameters)", 
            fontsize=16, fontweight='bold', ha='center', va='center', color='#0f172a')
    ax.text(7.5, 4.95, "Symmetric Funnel Design: 29 Features -> 8 Latent Dimensions (3.63x Compression) -> 29 Features", 
            fontsize=10.5, ha='center', va='center', color='#475569', style='italic')

    # Box styles
    b_in  = dict(boxstyle="round,pad=0.4", fc="#eff6ff", ec="#2563eb", lw=2)
    b_enc = dict(boxstyle="round,pad=0.4", fc="#f0f9ff", ec="#0284c7", lw=1.8)
    b_bot = dict(boxstyle="round,pad=0.5", fc="#faf5ff", ec="#7c3aed", lw=2.5)
    b_dec = dict(boxstyle="round,pad=0.4", fc="#fdf4ff", ec="#c026d3", lw=1.8)
    b_out = dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec="#16a34a", lw=2)

    y_pos = 2.8

    # 1. Input Layer
    ax.text(1.1, y_pos, "INPUT LAYER\n\n29 Features\n(V1..V28, Amount)\n\nZ-Scores\n(-inf, +inf)", 
            ha='center', va='center', fontsize=8.5, bbox=b_in, fontweight='bold', color="#1e3a8a")
    
    # Arrow
    ax.annotate('', xy=(2.3, y_pos), xytext=(1.9, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#2563eb"))
    
    # 2. Encoder Layer 1
    ax.text(3.2, y_pos, "ENCODER 1\n\nLinear(29 -> 20)\nBatchNorm1d(20)\nLeakyReLU(0.1)\n\n640 Params", 
            ha='center', va='center', fontsize=8.5, bbox=b_enc, color="#0369a1")

    # Arrow
    ax.annotate('', xy=(4.3, y_pos), xytext=(3.9, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#0284c7"))

    # 3. Encoder Layer 2
    ax.text(5.2, y_pos, "ENCODER 2\n\nLinear(20 -> 12)\nBatchNorm1d(12)\nLeakyReLU(0.1)\n\n276 Params", 
            ha='center', va='center', fontsize=8.5, bbox=b_enc, color="#0369a1")

    # Arrow
    ax.annotate('', xy=(6.3, y_pos), xytext=(5.9, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#7c3aed"))

    # 4. Latent Bottleneck (Z)
    ax.text(7.5, y_pos, "BOTTLENECK (z)\n\nLinear(12 -> 8)\nz in R^8\n\n8-Number Summary\n104 Params", 
            ha='center', va='center', fontsize=9.5, bbox=b_bot, fontweight='bold', color="#5b21b6")

    # Arrow
    ax.annotate('', xy=(8.7, y_pos), xytext=(8.3, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#c026d3"))

    # 5. Decoder Layer 1
    ax.text(9.8, y_pos, "DECODER 1\n\nLinear(8 -> 12)\nBatchNorm1d(12)\nLeakyReLU(0.1)\n\n132 Params", 
            ha='center', va='center', fontsize=8.5, bbox=b_dec, color="#86198f")

    # Arrow
    ax.annotate('', xy=(10.9, y_pos), xytext=(10.5, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#c026d3"))

    # 6. Decoder Layer 2
    ax.text(11.8, y_pos, "DECODER 2\n\nLinear(12 -> 20)\nBatchNorm1d(20)\nLeakyReLU(0.1)\n\n300 Params", 
            ha='center', va='center', fontsize=8.5, bbox=b_dec, color="#86198f")

    # Arrow
    ax.annotate('', xy=(12.9, y_pos), xytext=(12.5, y_pos),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#16a34a"))

    # 7. Output Layer
    ax.text(13.9, y_pos, "OUTPUT LAYER\n\nLinear(20 -> 29)\nReconstructed x^\n\nPure Linear\n609 Params", 
            ha='center', va='center', fontsize=8.5, bbox=b_out, fontweight='bold', color="#15803d")

    # Bottom notes
    note_box = patches.FancyBboxPatch((1.5, 0.4), 12.0, 0.8,
                                      boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1", lw=1.2)
    ax.add_patch(note_box)
    ax.text(7.5, 0.8, 
            "Hidden Activation: LeakyReLU (alpha = 0.1) prevents dead neurons  |  "
            "Output Layer: Linear (No Sigmoid) preserves real-valued Z-scores  |  "
            "Total Parameters: 2,061",
            ha='center', va='center', fontsize=9.5, color="#334155", fontweight='medium')

    plt.tight_layout()
    for p in ['results/figures/fig_ae_layer_architecture.png', 'notebooks/images/fig_ae_layer_architecture.png']:
        plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig_ae_layer_architecture.png successfully!")

if __name__ == '__main__':
    draw_layer_diagram()
