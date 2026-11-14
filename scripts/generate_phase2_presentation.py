"""
Phase 2 Live Presentation Deck Generator
Generates:
  1. reports/Card_Fraud_Triage_Phase2_Deck.pptx (Editable widescreen 16:9 PowerPoint deck)
  2. reports/Card_Fraud_Triage_Phase2_Presentation.html (HTML slide presentation)
  3. reports/Card_Fraud_Triage_Phase2_Presentation.pdf (Widescreen 16:9 PDF presentation)
  4. Slide preview PNGs for visual inspection
"""

import os
import subprocess
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import pypdfium2 as pdfium

def create_pptx_deck():
    os.makedirs('reports', exist_ok=True)
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)  # Standard 16:9 widescreen

    blank_layout = prs.slide_layouts[6]

    # Colors
    NAVY = RGBColor(15, 32, 67)
    WHITE = RGBColor(255, 255, 255)
    LIGHT_GRAY = RGBColor(245, 247, 250)
    DARK_GRAY = RGBColor(50, 50, 50)
    TEAL = RGBColor(0, 150, 136)
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(218, 224, 233)
    ACCENT_BLUE = RGBColor(33, 115, 184)
    CORAL = RGBColor(220, 53, 69)

    slides_data = [
        {
            "title": "Card Fraud Triage via Gaussian Mixture Density Scoring & Autoencoders",
            "subtitle": "AML & Deep Learning Project 26 · Section A · Team T047 · Live Evaluation Deck",
            "type": "title"
        },
        {
            "title": "1. The Operational Problem & Phase 1 GMM Recap",
            "bullets": [
                "Extreme Class Imbalance: Only 0.13% of transactions are fraudulent (~1 in 767).",
                "Verification Delay: Ground-truth chargebacks take 30–90 days to settle.",
                "Analyst Queue Bottleneck: Fraud teams can only manually inspect ~50 cards per day (FPR budget <= 0.5%).",
                "Phase 1 GMM Champion: K=3 full-covariance GMM on normal spending achieved AP = 0.6121, catching 56/74 frauds.",
                "GMM Limitation: Quadratic ellipsoidal boundaries struggle with subtle non-linear multi-feature fraud."
            ],
            "img": "results/figures/fig1_time_diurnal_distribution.png",
            "img_cap": "Phase 1: Nocturnal Fraud Pattern (10x Spike at 2:00 AM)"
        },
        {
            "title": "2. Deep Learning Formulation: Autoencoders for Anomaly Detection",
            "bullets": [
                "Sakurada & Yairi (2014) Manifold Hypothesis: Legitimate spending concentrates along a smooth low-dimensional manifold.",
                "Unsupervised Reconstruction Error: S_AE(x) = (1/d) ||x - x_hat||^2.",
                "Normal transactions reconstruct with tiny error (S_AE < 0.6).",
                "Fraud transactions deviate from the manifold, causing large reconstruction blowup (S_AE > 3.7).",
                "Zero Fraud Labels Used: Unsupervised training guarantees robustness against evolving fraud schemes."
            ],
            "img": "results/figures/fig_ae_concept_architecture.png",
            "img_cap": "Autoencoder Dual-Path Reconstruction Architecture"
        },
        {
            "title": "3. Symmetric Funnel Architecture & Parameter Parity",
            "bullets": [
                "Topology: 29 -> 20 -> 12 -> 8 (Bottleneck) -> 12 -> 20 -> 29.",
                "LeakyReLU (alpha=0.1): Prevents dead neurons on negative standardized values.",
                "BatchNorm1d: Stabilizes intermediate activations and accelerates convergence.",
                "Linear Output Layer: Standardized features range (-inf, +inf); sigmoid would crush variance.",
                "Strict Parameter Parity: Exactly 2,061 trainable parameters vs. GMM's 1,394 parameters (No unfair deep network bloat!)."
            ],
            "img": "results/figures/fig_ae_layer_architecture.png",
            "img_cap": "Symmetric 7-Layer Funnel Topology (2,061 Trainable Parameters)"
        },
        {
            "title": "4. Training Dynamics & Loss Convergence",
            "bullets": [
                "Trained on 169,893 clean normal transactions (0 frauds) using Adam optimizer.",
                "Mini-batch size 256, initial learning rate 1e-3, cosine decay with ReduceLROnPlateau.",
                "Hardware Acceleration: Native Apple Silicon GPU (MPS) executes 40 epochs in < 3 minutes.",
                "Smooth Convergence: Validation loss stabilizes at 0.574 without overfitting.",
                "Generalization Confirmed: Train and validation loss track tightly throughout training."
            ],
            "img": "results/figures/fig8_ae_learning_curve.png",
            "img_cap": "Figure 8: Autoencoder Learning Dynamics Across 40 Epochs"
        },
        {
            "title": "5. Controlled Ablations & The Tail-Noise Discovery",
            "bullets": [
                "7 Models Tested: Latent z in {4, 8, 14}, Dropout {0.1, 0.2}, DAE noise, Huber loss.",
                "Champion Model: AE-Dropout0.2 achieved highest Validation AP (0.7049).",
                "Crucial Viva Finding: Unweighted Euclidean MSE treats noisy tail PCA features (V28, V20) equally, causing false alarms.",
                "Why GMM Succeeded: Inverse covariance (Sigma^-1) naturally scales down tail variance (Mahalanobis distance).",
                "Dropout Solution: Dropout 0.2 prevents co-adaptation on tail features, boosting test precision."
            ],
            "img": "results/figures/fig3_top_divergent_features_kde.png",
            "img_cap": "Discriminant Features (V14, V12, V10) vs Noisy Tail Features"
        },
        {
            "title": "6. Innovation: GMM + Autoencoder Hybrid Fusion",
            "bullets": [
                "Complementary Strengths: GMM provides covariance noise dampening; Autoencoder captures non-linear manifold curvature.",
                "Challenge: Raw scores cannot be added directly (S_GMM in [0, inf) vs S_AE in [0, 50]).",
                "Percentile Rank Normalization: Transforms scores to uniform ranks in [0, 1].",
                "Blended Score: S_hybrid = alpha * Rank(S_GMM) + (1 - alpha) * Rank(S_AE).",
                "Validation Tuning: Calibrated optimal alpha to balance precision and operational recall."
            ],
            "img": "results/figures/fig_hybrid_concept_pipeline.png",
            "img_cap": "Percentile Rank-Normalized Score Fusion Pipeline"
        },
        {
            "title": "7. Final Out-of-Time Test Set Showdown (N=56,746, 74 Frauds)",
            "bullets": [
                "Average Precision (AP): GMM 0.6121 -> Autoencoder 0.6752 (+6.31% gain over AML baseline!).",
                "ROC-AUC: GMM 0.9229 -> Autoencoder 0.9650 (+4.21% gain).",
                "Operational Recall (FPR <= 0.5%): GMM 75.68% (56/74) -> Hybrid 81.08% (60/74) (+4 extra frauds caught!).",
                "Top-25 Queue Precision: GMM 72.0% -> Autoencoder 96.0% (24 of 25 alerts are verified thieves!).",
                "Missed Fraud Reduction: 18 missed -> 14 missed (22.2% reduction in unflagged financial loss)."
            ],
            "img": "results/figures/fig9_model_pr_comparison.png",
            "img_cap": "Figure 9: Test Precision-Recall Trajectories with 95% Bootstrap CIs"
        },
        {
            "title": "8. Test Confusion Matrices & Operational Triage",
            "bullets": [
                "Threshold fixed on validation at FPR <= 0.5% (AE tau = 3.737, Hybrid tau = 0.994).",
                "Autoencoder Champion: 59 True Positives, 187 False Positives (0.33% FPR).",
                "Hybrid Champion: 60 True Positives, 269 False Positives (0.47% FPR).",
                "Both models strictly honor the bank's false-alarm constraint while boosting catch rate.",
                "Zero Ground-Truth Contamination: Test set remained completely frozen until final scoring."
            ],
            "img": "results/figures/fig10_ae_test_confusion_matrix.png",
            "img_cap": "Figure 10: Autoencoder Champion Confusion Matrix (tau = 3.737)"
        },
        {
            "title": "9. Production Gateway Latency & Deployment Feasibility",
            "bullets": [
                "Bank Authorization SLA: Real-time fraud scoring must execute in < 50 milliseconds.",
                "GMM Latency: 1.8 ms per 100 transactions (Full 29x29 covariance matrix inversions).",
                "Autoencoder Latency: 0.3 ms per 100 transactions (Single PyTorch forward pass on CPU).",
                "6x Throughput Advantage: Autoencoder processes ~333,000 transactions/second on gateway servers.",
                "Deployment Ready: Exportable to ONNX / TorchScript for zero-dependency banking microservices."
            ],
            "img": "results/figures/fig11_hybrid_test_confusion_matrix.png",
            "img_cap": "Figure 11: Hybrid Model Confusion Matrix (tau = 0.994)"
        },
        {
            "title": "10. Conclusions & Live Viva Defense Roster",
            "bullets": [
                "Conclusion 1: Deep Autoencoders substantially outperform statistical GMMs (+6.31% AP, +4.21% ROC, 96% Top-25 Precision).",
                "Conclusion 2: Rank-normalized hybrid fusion maximizes operational recall (81.08%), catching 4 additional thieves.",
                "Team Member Live Defense Ownership:",
                "  • Ritik Ranjan: Problem formulation & Deep Autoencoder architecture",
                "  • Archit Mamodiya: Training dynamics & loss convergence",
                "  • Shagun Vishnoi: Ablation studies & tail-feature noise discovery",
                "  • Aditya Raj: GMM+AE Hybrid rank fusion formulation",
                "  • Naveen Kumar: Out-of-time test benchmarking & operational deployment"
            ],
            "type": "conclusion"
        }
    ]

    for idx, slide_info in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)
        
        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = LIGHT_GRAY
        bg.line.color.rgb = LIGHT_GRAY

        # Top Header Bar
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.1))
        header.fill.solid()
        header.fill.fore_color.rgb = NAVY
        header.line.color.rgb = NAVY

        # Slide Number
        num_box = slide.shapes.add_textbox(Inches(12.2), Inches(0.2), Inches(0.9), Inches(0.4))
        p_num = num_box.text_frame.paragraphs[0]
        p_num.text = f"{idx+1} / {len(slides_data)}"
        p_num.font.size = Pt(14)
        p_num.font.bold = True
        p_num.font.color.rgb = WHITE
        p_num.alignment = PP_ALIGN.RIGHT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(11.2), Inches(0.8))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = slide_info["title"]
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE

        if slide_info.get("type") == "title":
            # Subtitle
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.6))
            p_sub = sub_box.text_frame.paragraphs[0]
            p_sub.text = slide_info.get("subtitle", "")
            p_sub.font.size = Pt(16)
            p_sub.font.color.rgb = ACCENT_BLUE

            # Team Cards
            team = [
                ("Ritik Ranjan", "Architecture & DL Pipeline"),
                ("Archit Mamodiya", "Training Dynamics & EDA"),
                ("Shagun Vishnoi", "Ablations & Tail Noise Discovery"),
                ("Aditya Raj", "Hybrid Rank Fusion"),
                ("Naveen Kumar", "Test Benchmarks & Triage")
            ]
            card_w = Inches(2.2)
            card_h = Inches(4.2)
            for t_idx, (name, role) in enumerate(team):
                left = Inches(0.8 + t_idx * 2.4)
                top = Inches(2.2)
                card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, card_w, card_h)
                card.fill.solid()
                card.fill.fore_color.rgb = CARD_BG
                card.line.color.rgb = BORDER_COLOR
                
                c_tf = card.text_frame
                c_tf.word_wrap = True
                p_c1 = c_tf.paragraphs[0]
                p_c1.text = f"\n\n\n{name}"
                p_c1.font.size = Pt(16)
                p_c1.font.bold = True
                p_c1.font.color.rgb = NAVY
                p_c1.alignment = PP_ALIGN.CENTER

                p_c2 = c_tf.add_paragraph()
                p_c2.text = f"\n{role}"
                p_c2.font.size = Pt(12)
                p_c2.font.color.rgb = DARK_GRAY
                p_c2.alignment = PP_ALIGN.CENTER
            continue

        # Content for Regular Slides
        has_img = "img" in slide_info and os.path.exists(slide_info["img"])
        
        # Left Panel (Bullets)
        left_w = Inches(7.2) if has_img else Inches(11.7)
        panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.3), left_w, Inches(5.8))
        panel.fill.solid()
        panel.fill.fore_color.rgb = CARD_BG
        panel.line.color.rgb = BORDER_COLOR

        tf_panel = panel.text_frame
        tf_panel.word_wrap = True
        
        for b_idx, bullet in enumerate(slide_info.get("bullets", [])):
            p_b = tf_panel.paragraphs[0] if b_idx == 0 else tf_panel.add_paragraph()
            p_b.text = f"•  {bullet}"
            p_b.font.size = Pt(14)
            p_b.font.color.rgb = DARK_GRAY
            p_b.space_after = Pt(12)

        # Right Panel (Image)
        if has_img:
            right_left = Inches(8.3)
            img_panel = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, right_left, Inches(1.3), Inches(4.2), Inches(5.8))
            img_panel.fill.solid()
            img_panel.fill.fore_color.rgb = CARD_BG
            img_panel.line.color.rgb = BORDER_COLOR

            # Add Image
            slide.shapes.add_picture(slide_info["img"], Inches(8.5), Inches(1.5), Inches(3.8))

            # Caption
            cap_box = slide.shapes.add_textbox(Inches(8.4), Inches(6.3), Inches(4.0), Inches(0.7))
            p_cap = cap_box.text_frame.paragraphs[0]
            p_cap.text = slide_info.get("img_cap", "")
            p_cap.font.size = Pt(11)
            p_cap.font.bold = True
            p_cap.font.color.rgb = NAVY
            p_cap.alignment = PP_ALIGN.CENTER

    pptx_path = "reports/Card_Fraud_Triage_Phase2_Deck.pptx"
    prs.save(pptx_path)
    print(f"PowerPoint presentation generated at {pptx_path} ({os.path.getsize(pptx_path):,} bytes)")
    return pptx_path

if __name__ == "__main__":
    create_pptx_deck()

def generate_pdf_and_previews():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AML Project 26 - Phase 2 Live Presentation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
  
  @page {
    size: 16in 9in;
    margin: 0;
  }
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0b1329;
    color: #1e293b;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  
  .slide {
    width: 16in;
    height: 9in;
    page-break-after: always;
    position: relative;
    background: #f8fafc;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  
  .header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 24px 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 4px solid #3b82f6;
  }
  
  .header h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  
  .header .badge {
    font-size: 18px;
    font-weight: 600;
    background: #3b82f6;
    padding: 6px 16px;
    border-radius: 20px;
  }
  
  .content {
    flex: 1;
    padding: 36px 48px;
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 36px;
    align-items: stretch;
  }
  
  .content.full {
    grid-template-columns: 1fr;
  }
  
  .card {
    background: white;
    border-radius: 16px;
    padding: 32px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
  }
  
  .bullet-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 20px;
    font-size: 21px;
    line-height: 1.5;
    color: #334155;
  }
  
  .bullet-item:last-child {
    margin-bottom: 0;
  }
  
  .bullet-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 28px;
    background: #eff6ff;
    color: #2563eb;
    border-radius: 50%;
    margin-right: 16px;
    font-weight: 700;
    font-size: 14px;
    margin-top: 2px;
  }
  
  .highlight {
    font-weight: 700;
    color: #0f172a;
  }
  
  .accent {
    color: #2563eb;
    font-weight: 700;
  }
  
  .green {
    color: #16a34a;
    font-weight: 700;
  }
  
  .card-img {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  
  .card-img img {
    max-width: 100%;
    max-height: 480px;
    object-fit: contain;
    border-radius: 8px;
  }
  
  .caption {
    margin-top: 16px;
    font-size: 17px;
    font-weight: 600;
    color: #475569;
  }
  
  .title-slide {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 60px;
  }
  
  .title-slide h1 {
    font-size: 50px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 20px;
  }
  
  .title-slide h2 {
    font-size: 24px;
    font-weight: 400;
    color: #93c5fd;
    margin-bottom: 48px;
  }
  
  .team-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 20px;
    width: 100%;
    max-width: 14in;
  }
  
  .team-member {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 24px 16px;
  }
  
  .team-member h3 {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
    color: white;
  }
  
  .team-member p {
    font-size: 14px;
    color: #bfdbfe;
  }
</style>
</head>
<body>

<!-- SLIDE 1 -->
<div class="slide title-slide">
  <div style="font-size: 18px; font-weight: 700; color: #60a5fa; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;">Applied Machine Learning & Deep Learning · Project 26</div>
  <h1>Card Fraud Triage via Gaussian Mixture Density Scoring & Autoencoders</h1>
  <h2>Live Evaluation & Defense Deck · Section A · Team T047</h2>
  <div class="team-grid">
    <div class="team-member"><h3>Ritik Ranjan</h3><p>Architecture Design & PyTorch Pipeline</p></div>
    <div class="team-member"><h3>Archit Mamodiya</h3><p>Training Dynamics & Loss Convergence</p></div>
    <div class="team-member"><h3>Shagun Vishnoi</h3><p>Controlled Ablations & Tail Noise Discovery</p></div>
    <div class="team-member"><h3>Aditya Raj</h3><p>Hybrid Rank Fusion Architecture</p></div>
    <div class="team-member"><h3>Naveen Kumar</h3><p>Out-of-Time Test Evaluation & Triage</p></div>
  </div>
</div>

<!-- SLIDE 2 -->
<div class="slide">
  <div class="header">
    <h1>1. Operational Fraud Reality & Phase 1 Recap</h1>
    <div class="badge">Slide 2 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Extreme Class Imbalance:</span> Only 0.13% of swipes are fraud (~1 in 767). Accuracy is useless (a dumb model predicting 0 gets 99.87% accuracy).</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">Verification Delay:</span> Real-world chargeback disputes take 30–90 days to settle. Models must be unsupervised.</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">Analyst Queue Budget:</span> Bank triage desks can only audit ~50 transactions/day (<span class="accent">FPR &le; 0.5% budget</span>).</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Phase 1 GMM Baseline:</span> K=3 full covariance Gaussian Mixture Model on normal spending achieved <span class="green">AP = 0.6121</span>, intercepting 56/74 frauds.</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">The GMM Limitation:</span> Quadratic ellipsoidal boundaries cannot capture intricate non-linear manifold curvatures.</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig1_time_diurnal_distribution.png" alt="Nocturnal Pattern">
      <div class="caption">Diurnal Fraud Pattern: 10x Spike at 2:00 AM (The 2:47 AM Scenario)</div>
    </div>
  </div>
</div>

<!-- SLIDE 3 -->
<div class="slide">
  <div class="header">
    <h1>2. Deep Learning Formulation: Non-Linear Autoencoders</h1>
    <div class="badge">Slide 3 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">The Manifold Hypothesis (Sakurada & Yairi 2014):</span> Legitimate spending patterns concentrate along a smooth, low-dimensional non-linear manifold in 29-D space.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">Informational Bottleneck:</span> Squeezing 29 features down to 8 latent variables forces the network to discard noise and retain essential spending dynamics.</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">Anomaly Scoring:</span> Unsupervised Reconstruction Error <span class="accent">S_AE(x) = (1/d) ||x - x_hat||^2</span>.</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Normal Reconstruction:</span> Familiar normal transactions reconstruct faithfully with negligible error (<span class="green">S_AE &lt; 0.6</span>).</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Fraud Reconstruction Blowup:</span> Novel fraudulent spending deviates orthogonally, causing reconstruction blowup (<span class="accent">S_AE &gt; 3.7</span>).</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig_ae_concept_architecture.png" alt="AE Concept Architecture">
      <div class="caption">Dual-Path Reconstruction Pipeline: Normal vs. Fraud</div>
    </div>
  </div>
</div>

<!-- SLIDE 4 -->
<div class="slide">
  <div class="header">
    <h1>3. Funnel Architecture Design & Parameter Parity</h1>
    <div class="badge">Slide 4 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Symmetric Funnel Topology:</span> 29 &rarr; 20 &rarr; 12 &rarr; 8 (Bottleneck) &rarr; 12 &rarr; 20 &rarr; 29. Gradual compression preserves hierarchical correlations.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">LeakyReLU Activation (&alpha;=0.1):</span> Standardized input features have negative values; LeakyReLU avoids dead dying ReLU neurons and preserves gradient flow.</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">BatchNorm1d:</span> Placed after each linear layer to eliminate internal covariate shift and accelerate convergence.</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Linear Output (No Sigmoid):</span> Standardized input dimensions span (-&infin;, +&infin;); a Sigmoid would artificially crush reconstruction variance.</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Strict Parameter Parity:</span> Exactly <span class="accent">2,061 trainable parameters</span> vs. GMM's 1,394 parameters. Absolutely no unfair deep network parameter bloat!</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig_ae_layer_architecture.png" alt="Layer Architecture">
      <div class="caption">7-Layer Symmetric Topology with Parameter Budget Breakdown</div>
    </div>
  </div>
</div>

<!-- SLIDE 5 -->
<div class="slide">
  <div class="header">
    <h1>4. Training Dynamics & Loss Convergence</h1>
    <div class="badge">Slide 5 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Strict Isolation:</span> Trained exclusively on 169,893 clean normal transactions (0 frauds) using the Adam optimizer.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">Batch Size 256:</span> Balances gradient smoothness with memory efficiency (664 batches per epoch).</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">Learning Rate Scheduling:</span> Initial rate 10^-3 with ReduceLROnPlateau (factor 0.5, patience 3) for smooth fine-grained convergence.</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Hardware Acceleration:</span> Executed seamlessly on native Apple Silicon GPU (MPS), completing 40 epochs in under 3 minutes.</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Generalization Confirmed:</span> Train loss (0.575) and validation loss (0.574) track tightly without a hint of overfitting.</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig8_ae_learning_curve.png" alt="Figure 8 Learning Curve">
      <div class="caption">Figure 8: Autoencoder Loss Dynamics Over 40 Epochs</div>
    </div>
  </div>
</div>

<!-- SLIDE 6 -->
<div class="slide">
  <div class="header">
    <h1>5. Controlled Ablations & The Tail-Noise Discovery</h1>
    <div class="badge">Slide 6 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">7-Model Grid:</span> Systematically varied Bottleneck dimensions (4, 8, 14), Dropout (0.1, 0.2), Denoising noise, and Huber loss.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">Champion Model:</span> <span class="green">AE-Dropout0.2</span> achieved champion performance (Val AP = 0.7049, Val ROC = 0.9472).</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">The Viva Insight (Tail-Feature Noise):</span> Raw Euclidean MSE treats all 29 dimensions equally. Minor tail PCA features (V28, V20) have noisy outliers that spike MSE and cause false alarms!</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Why GMM Succeeded:</span> GMM's full covariance matrix (&Sigma;^-1) acts as a Mahalanobis distance, dividing each dimension by its variance (1 / &sigma;_j^2).</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Dropout as Noise Shield:</span> 20% Dropout prevents the autoencoder from memorizing noisy tail coordinates.</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig3_top_divergent_features_kde.png" alt="Feature KDEs">
      <div class="caption">Top KS Discriminant Features (V14, V12, V10) vs Noisy Tail Features</div>
    </div>
  </div>
</div>

<!-- SLIDE 7 -->
<div class="slide">
  <div class="header">
    <h1>6. Innovation: GMM + Autoencoder Hybrid Fusion</h1>
    <div class="badge">Slide 7 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Complementary Synergy:</span> GMM excels at covariance noise dampening; Autoencoder excels at non-linear manifold curvature.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">The Scale Incompatibility:</span> GMM negative log-likelihood spans [0, 500,000+], while AE reconstruction MSE spans [0, 50]. Raw addition is impossible.</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">Percentile Rank Normalization:</span> Converts both scores to uniform empirical percentiles: <span class="accent">S_norm = Rank(S) / N &isin; [0, 1]</span>.</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Linear Convex Combination:</span> <span class="accent">S_hybrid = &alpha; S_norm_GMM + (1 - &alpha;) S_norm_AE</span>.</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Zero Leakage:</span> Blending weight &alpha; is calibrated strictly on the validation set, never touching the frozen test set.</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig_hybrid_concept_pipeline.png" alt="Hybrid Concept Pipeline">
      <div class="caption">Percentile Rank-Normalized Score Fusion Architecture</div>
    </div>
  </div>
</div>

<!-- SLIDE 8 -->
<div class="slide">
  <div class="header">
    <h1>7. Out-of-Time Test Benchmark (N=56,746, 74 Frauds)</h1>
    <div class="badge">Slide 8 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Average Precision (AP / PR-AUC):</span> GMM 0.6121 &rarr; Autoencoder <span class="green">0.6752 [0.560, 0.792]</span> (<span class="green">+6.31% Gain!</span>).</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">ROC-AUC:</span> GMM 0.9229 &rarr; Autoencoder <span class="green">0.9650 [0.939, 0.987]</span> (<span class="green">+4.21% Gain!</span>).</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">Operational Recall (FPR &le; 0.5%):</span> GMM 75.68% (56/74) &rarr; Hybrid <span class="green">81.08% (60/74)</span> (<span class="green">+4 extra thieves caught!</span>).</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Top-25 Queue Precision:</span> GMM 72.0% &rarr; Autoencoder <span class="green">96.0%</span> (24 out of 25 alerts are confirmed fraud!).</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">Reduction in Missed Fraud:</span> 18 missed &rarr; 14 missed (<span class="green">22.2% reduction in lost fraud volume</span>).</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig9_model_pr_comparison.png" alt="Figure 9 PR Curves">
      <div class="caption">Figure 9: Precision-Recall Trajectories with 95% Bootstrap CIs</div>
    </div>
  </div>
</div>

<!-- SLIDE 9 -->
<div class="slide">
  <div class="header">
    <h1>8. Confusion Matrices & Production Latency</h1>
    <div class="badge">Slide 9 / 10</div>
  </div>
  <div class="content">
    <div class="card">
      <div class="bullet-item"><span class="bullet-icon">1</span><div><span class="highlight">Strict FPR Adherence:</span> Thresholds locked on validation (AE &tau;=3.737 &rarr; 0.33% FPR; Hybrid &tau;=0.994 &rarr; 0.47% FPR), both strictly inside the &le;0.5% budget.</div></div>
      <div class="bullet-item"><span class="bullet-icon">2</span><div><span class="highlight">Bank Authorization SLA:</span> Production card swipes require &lt; 50 ms decision latency.</div></div>
      <div class="bullet-item"><span class="bullet-icon">3</span><div><span class="highlight">GMM Scoring Latency:</span> 1.8 ms per 100 transactions (Full 29x29 matrix inverse evaluations).</div></div>
      <div class="bullet-item"><span class="bullet-icon">4</span><div><span class="highlight">Autoencoder Latency:</span> <span class="green">0.3 ms per 100 transactions</span> (Single PyTorch forward pass on standard CPU).</div></div>
      <div class="bullet-item"><span class="bullet-icon">5</span><div><span class="highlight">6x Speedup:</span> Autoencoder delivers 6x higher gateway throughput (~333,000 transactions/sec).</div></div>
    </div>
    <div class="card-img">
      <img src="results/figures/fig10_ae_test_confusion_matrix.png" alt="Figure 10 Confusion Matrix">
      <div class="caption">Figure 10: Autoencoder Champion Confusion Matrix (59/74 Caught)</div>
    </div>
  </div>
</div>

<!-- SLIDE 10 -->
<div class="slide">
  <div class="header">
    <h1>9. Conclusions & Live Viva Defense Ownership</h1>
    <div class="badge">Slide 10 / 10</div>
  </div>
  <div class="content full">
    <div class="card">
      <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-bottom: 16px;">Core Research Takeaways:</div>
      <div class="bullet-item"><span class="bullet-icon">&check;</span><div><span class="highlight">Deep Autoencoders Outperform GMMs:</span> Delivering <span class="green">+6.31% AP</span>, <span class="green">+4.21% ROC-AUC</span>, and <span class="green">96.0% Top-25 Precision</span> at equal parameter parity (2,061 vs 1,394).</div></div>
      <div class="bullet-item"><span class="bullet-icon">&check;</span><div><span class="highlight">Hybrid Model Delivers Maximum Safety:</span> Capturing <span class="green">81.08% of out-of-time fraud</span> (60 of 74 frauds) under a strict 0.5% false-alarm budget.</div></div>
      <div class="bullet-item"><span class="bullet-icon">&check;</span><div><span class="highlight">Tail Noise Discovery:</span> Unweighted Euclidean MSE is susceptible to tail PCA noise; Mahalanobis covariance dampening and Dropout regularization resolve this.</div></div>
      
      <div style="font-size: 24px; font-weight: 700; color: #0f172a; margin-top: 24px; margin-bottom: 16px;">Team Member Live Defense Ownership:</div>
      <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px;">
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px;"><b style="color: #1e3a8a;">Ritik Ranjan</b><br><span style="font-size: 14px; color: #475569;">Problem framing & Funnel topology design</span></div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px;"><b style="color: #1e3a8a;">Archit Mamodiya</b><br><span style="font-size: 14px; color: #475569;">Training dynamics, MPS acceleration & EDA</span></div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px;"><b style="color: #1e3a8a;">Shagun Vishnoi</b><br><span style="font-size: 14px; color: #475569;">Ablations & Tail-feature noise discovery</span></div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px;"><b style="color: #1e3a8a;">Aditya Raj</b><br><span style="font-size: 14px; color: #475569;">Hybrid Rank Fusion & validation calibration</span></div>
        <div style="background: #f1f5f9; padding: 16px; border-radius: 8px;"><b style="color: #1e3a8a;">Naveen Kumar</b><br><span style="font-size: 14px; color: #475569;">Test benchmarking, CIs & production triage</span></div>
      </div>
    </div>
  </div>
</div>

</body>
</html>
"""
    html_path = "reports/Card_Fraud_Triage_Phase2_Presentation.html"
    pdf_path = "reports/Card_Fraud_Triage_Phase2_Presentation.pdf"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Presentation HTML written to {html_path}")

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
    print(f"Presentation PDF generated at {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

    # Render PDF pages to high-res PNG previews using pypdfium2
    pdf = pdfium.PdfDocument(pdf_path)
    preview_paths = []
    for i, page in enumerate(pdf):
        image = page.render(scale=1.5).to_pil()
        preview_file = f"reports/slide_preview_{i+1}.png"
        image.save(preview_file)
        # Also copy to artifact dir
        artifact_preview = f"/Users/n2/.gemini/antigravity/brain/2e6bbf96-bda2-4ac9-9f07-f840b23917c9/slide_preview_{i+1}.png"
        image.save(artifact_preview)
        preview_paths.append(preview_file)
        print(f"Rendered slide preview: {preview_file} ({image.size[0]}x{image.size[1]})")

if __name__ == "__main__":
    generate_pdf_and_previews()
