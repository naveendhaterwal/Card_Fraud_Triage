"""
Phase 2 Deep Learning Master Experiment Runner
Trains baseline AE, executes ablation grid, fits hybrid model, and logs all test metrics.
"""

import os
import json
import joblib
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

from src.data import load_and_preprocess_data, get_dataloaders
from src.models import Autoencoder, DenoisingAutoencoder, compute_reconstruction_error
from src.training import train_autoencoder
from src.hybrid import compute_hybrid_score, optimize_hybrid_alpha
from src.evaluation import calibrate_threshold_at_fpr, evaluate_model_performance


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline():
    set_seed(42)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    # Ensure result directories exist
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Load Data
    print("\n[Step 1] Loading and preprocessing datasets...")
    data = load_and_preprocess_data()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_val, y_test = data['y_val'], data['y_test']
    
    print(f"Train Legit: {X_train.shape[0]:,} samples | Val Mixed: {X_val.shape[0]:,} ({np.sum(y_val)} frauds) | Test Mixed: {X_test.shape[0]:,} ({np.sum(y_test)} frauds)")
    
    train_loader, val_loader = get_dataloaders(X_train, X_val, batch_size=256)
    
    # 2. Train Champion Baseline Autoencoder (Latent Dim = 8)
    print("\n[Step 2] Training Baseline Autoencoder (Latent Dim = 8)...")
    baseline_ae = Autoencoder(input_dim=29, latent_dim=8, hidden_dims=(20, 12), dropout_rate=0.0)
    baseline_ae, history_baseline = train_autoencoder(
        baseline_ae, train_loader, val_loader, X_val, y_val,
        epochs=35, lr=1e-3, weight_decay=1e-5, patience=7, device=device, verbose=True
    )
    
    # Save Baseline Model & History
    torch.save(baseline_ae.state_dict(), 'results/models/ae_baseline_latent8.pt')
    with open('results/metrics/ae_baseline_history.json', 'w') as f:
        json.dump(history_baseline, f, indent=4)
        
    # Plot Training Loss Curves
    plt.figure(figsize=(9, 4.5))
    plt.plot(history_baseline['epoch'], history_baseline['train_loss'], label='Train Loss (Legitimate)', color='#1f77b4', lw=2)
    plt.plot(history_baseline['epoch'], history_baseline['val_loss'], label='Validation Loss (Mixed)', color='#ff7f0e', lw=2, linestyle='--')
    plt.xlabel('Epoch', fontsize=11, fontweight='bold')
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=11, fontweight='bold')
    plt.title('Figure 8: Autoencoder Learning Dynamics & Convergence', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig('results/figures/fig8_ae_learning_curve.png', dpi=300)
    plt.close()
    print("Saved results/figures/fig8_ae_learning_curve.png")
    
    # 3. Ablation Studies (Controlled Experiments)
    print("\n[Step 3] Executing Controlled Ablation Studies...")
    ablations = [
        # (name, model_class, latent_dim, dropout, noise, loss_type)
        ("AE-Latent4", Autoencoder, 4, 0.0, 0.0, 'mse'),
        ("AE-Latent8-Champion", Autoencoder, 8, 0.0, 0.0, 'mse'),
        ("AE-Latent14", Autoencoder, 14, 0.0, 0.0, 'mse'),
        ("AE-Dropout0.1", Autoencoder, 8, 0.1, 0.0, 'mse'),
        ("AE-Dropout0.2", Autoencoder, 8, 0.2, 0.0, 'mse'),
        ("DAE-Noise0.1", DenoisingAutoencoder, 8, 0.0, 0.1, 'mse'),
        ("AE-HuberLoss", Autoencoder, 8, 0.0, 0.0, 'huber')
    ]
    
    ablation_summary = []
    trained_models = {"AE-Latent8-Champion": baseline_ae}
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    
    for name, m_class, lat_dim, drop, noise, l_type in ablations:
        print(f"\n--- Running Ablation: {name} ---")
        if name == "AE-Latent8-Champion":
            model = baseline_ae
            best_val_loss = min(history_baseline['val_loss'])
        else:
            if m_class == DenoisingAutoencoder:
                model = m_class(input_dim=29, latent_dim=lat_dim, hidden_dims=(20, 12), dropout_rate=drop, noise_factor=noise)
            else:
                model = m_class(input_dim=29, latent_dim=lat_dim, hidden_dims=(20, 12), dropout_rate=drop)
                
            model, hist = train_autoencoder(
                model, train_loader, val_loader, X_val, y_val,
                epochs=30, lr=1e-3, weight_decay=1e-5, patience=6, device=device, loss_type=l_type, verbose=False
            )
            best_val_loss = min(hist['val_loss'])
            trained_models[name] = model
            
        # Evaluate on Validation
        val_errors = compute_reconstruction_error(model, X_val_t, device=device, loss_type=l_type)
        val_ap = float(np.round(eval_metric := average_precision_score(y_val, val_errors), 4))
        
        ablation_summary.append({
            'experiment': name,
            'latent_dim': lat_dim,
            'dropout': drop,
            'noise': noise,
            'loss_type': l_type,
            'val_loss': float(np.round(best_val_loss, 5)),
            'val_ap': val_ap
        })
        print(f"--> {name}: Val Loss = {best_val_loss:.5f} | Val AP = {val_ap:.4f}")
        
    with open('results/metrics/ae_ablation_summary.json', 'w') as f:
        json.dump(ablation_summary, f, indent=4)
        
    # Select Champion Autoencoder based on Validation AP
    best_ablation = max(ablation_summary, key=lambda x: x['val_ap'])
    champion_name = best_ablation['experiment']
    champion_ae = trained_models[champion_name]
    print(f"\nChampion AE Selected: {champion_name} (Val AP = {best_ablation['val_ap']})")
    torch.save(champion_ae.state_dict(), 'results/models/ae_champion.joblib')
    
    # 4. Load GMM Champion and Compute Validation Scores for Hybrid Model
    print("\n[Step 4] Formulating GMM + AE Hybrid Innovation...")
    gmm_model = joblib.load('results/models/gmm_champion.joblib')
    val_gmm_scores = -gmm_model.score_samples(X_val)
    val_ae_scores = compute_reconstruction_error(champion_ae, X_val_t, device=device, loss_type='mse')
    
    best_alpha, best_val_hybrid_ap, search_results = optimize_hybrid_alpha(
        val_gmm_scores, val_ae_scores, y_val, norm_method='rank'
    )
    print(f"Optimal Hybrid Weight: alpha = {best_alpha} (GMM: {best_alpha:.2f}, AE: {1-best_alpha:.2f})")
    print(f"Validation AP: GMM = {average_precision_score(y_val, val_gmm_scores):.4f} | AE = {average_precision_score(y_val, val_ae_scores):.4f} | Hybrid = {best_val_hybrid_ap:.4f}")
    
    with open('results/metrics/hybrid_validation_alpha_grid.json', 'w') as f:
        json.dump({'best_alpha': best_alpha, 'best_val_ap': best_val_hybrid_ap, 'grid': search_results}, f, indent=4)
        
    # 5. Calibrate Operational Thresholds on Validation Set (target FPR <= 0.5%)
    print("\n[Step 5] Calibrating Operational Thresholds at <= 0.5% FPR on Validation Data...")
    tau_gmm = calibrate_threshold_at_fpr(val_gmm_scores, y_val, target_fpr=0.005)
    tau_ae = calibrate_threshold_at_fpr(val_ae_scores, y_val, target_fpr=0.005)
    
    val_hybrid_scores = compute_hybrid_score(val_gmm_scores, val_ae_scores, alpha=best_alpha, norm_method='rank')
    tau_hybrid = calibrate_threshold_at_fpr(val_hybrid_scores, y_val, target_fpr=0.005)
    
    print(f"Calibrated Thresholds: tau_GMM = {tau_gmm:.2f} | tau_AE = {tau_ae:.4f} | tau_Hybrid = {tau_hybrid:.4f}")
    
    # 6. Single Final Evaluation on Frozen Out-Of-Time Test Set
    print("\n[Step 6] Running Final Out-of-Time Test Set Evaluation (56,746 samples, 74 frauds)...")
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    test_gmm_scores = -gmm_model.score_samples(X_test)
    test_ae_scores = compute_reconstruction_error(champion_ae, X_test_t, device=device, loss_type='mse')
    test_hybrid_scores = compute_hybrid_score(test_gmm_scores, test_ae_scores, alpha=best_alpha, norm_method='rank')
    
    # Compute Metrics with Bootstrap CIs
    metrics_gmm = evaluate_model_performance(y_test, test_gmm_scores, tau_gmm, model_name="GMM (K=3, Full Cov)")
    metrics_ae = evaluate_model_performance(y_test, test_ae_scores, tau_ae, model_name=f"Autoencoder ({champion_name})")
    metrics_hybrid = evaluate_model_performance(y_test, test_hybrid_scores, tau_hybrid, model_name=f"GMM + AE Hybrid (alpha={best_alpha})")
    
    with open('results/metrics/ae_test_metrics.json', 'w') as f:
        json.dump(metrics_ae, f, indent=4)
    with open('results/metrics/hybrid_test_metrics.json', 'w') as f:
        json.dump(metrics_hybrid, f, indent=4)
        
    print("\n" + "=" * 80)
    print("FINAL OUT-OF-TIME TEST BENCHMARK (N=56,746 | Frauds=74)")
    print("=" * 80)
    header = f"{'Metric':<24} | {'GMM Baseline':<16} | {'Autoencoder':<16} | {'GMM+AE Hybrid':<16}"
    print(header)
    print("-" * 80)
    print(f"{'Average Precision (AP)':<24} | {metrics_gmm['test_average_precision']:<16.4f} | {metrics_ae['test_average_precision']:<16.4f} | {metrics_hybrid['test_average_precision']:<16.4f}")
    print(f"{'  95% Bootstrap CI':<24} | [{metrics_gmm['test_average_precision_95ci'][0]:.3f}, {metrics_gmm['test_average_precision_95ci'][1]:.3f}]  | [{metrics_ae['test_average_precision_95ci'][0]:.3f}, {metrics_ae['test_average_precision_95ci'][1]:.3f}]  | [{metrics_hybrid['test_average_precision_95ci'][0]:.3f}, {metrics_hybrid['test_average_precision_95ci'][1]:.3f}]")
    print(f"{'ROC-AUC':<24} | {metrics_gmm['test_roc_auc']:<16.4f} | {metrics_ae['test_roc_auc']:<16.4f} | {metrics_hybrid['test_roc_auc']:<16.4f}")
    print(f"{'Recall @ <=0.5% FPR':<24} | {metrics_gmm['recall_at_0.5pct_fpr']*100:<15.1f}% | {metrics_ae['recall_at_0.5pct_fpr']*100:<15.1f}% | {metrics_hybrid['recall_at_0.5pct_fpr']*100:<15.1f}%")
    print(f"{'True Positives (Caught)':<24} | {metrics_gmm['true_positives']:<16d} | {metrics_ae['true_positives']:<16d} | {metrics_hybrid['true_positives']:<16d}")
    print(f"{'False Positives (Alarms)':<24} | {metrics_gmm['false_positives']:<16d} | {metrics_ae['false_positives']:<16d} | {metrics_hybrid['false_positives']:<16d}")
    print(f"{'False Negatives (Missed)':<24} | {metrics_gmm['false_negatives']:<16d} | {metrics_ae['false_negatives']:<16d} | {metrics_hybrid['false_negatives']:<16d}")
    print(f"{'Precision @ Top 50':<24} | {metrics_gmm['precision_top_50']*100:<15.1f}% | {metrics_ae['precision_top_50']*100:<15.1f}% | {metrics_hybrid['precision_top_50']*100:<15.1f}%")
    print("=" * 80)
    
    # 7. Generate Comparative Publication Figures
    print("\n[Step 7] Generating Comparison Figures...")
    
    # Figure 9: Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6))
    p_gmm, r_gmm, _ = precision_recall_curve(y_test, test_gmm_scores)
    p_ae, r_ae, _ = precision_recall_curve(y_test, test_ae_scores)
    p_hyb, r_hyb, _ = precision_recall_curve(y_test, test_hybrid_scores)
    
    plt.plot(r_gmm, p_gmm, label=f"GMM Champion (AP = {metrics_gmm['test_average_precision']:.3f})", color='#1f77b4', lw=2)
    plt.plot(r_ae, p_ae, label=f"Autoencoder (AP = {metrics_ae['test_average_precision']:.3f})", color='#2ca02c', lw=2)
    plt.plot(r_hyb, p_hyb, label=f"GMM+AE Hybrid (AP = {metrics_hybrid['test_average_precision']:.3f})", color='#d62728', lw=2.5)
    plt.axhline(y=74/56746, color='gray', linestyle=':', label='Random Guessing (Prevalence: 0.13%)')
    
    plt.xlabel('Recall (Fraction of Frauds Caught)', fontsize=11, fontweight='bold')
    plt.ylabel('Precision (Fraud Alert Accuracy)', fontsize=11, fontweight='bold')
    plt.title('Figure 9: Precision-Recall Trajectories: GMM vs AE vs Hybrid', fontsize=12, fontweight='bold')
    plt.legend(frameon=True, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/fig9_model_pr_comparison.png', dpi=300)
    plt.close()
    print("Saved results/figures/fig9_model_pr_comparison.png")
    
    # Figure 10: Autoencoder Test Confusion Matrix Heatmap
    cm_ae = np.array([[metrics_ae['true_negatives'], metrics_ae['false_positives']],
                      [metrics_ae['false_negatives'], metrics_ae['true_positives']]])
    plt.figure(figsize=(7, 5.5))
    annot_matrix = [
        [f"TN\n{metrics_ae['true_negatives']:,}\n(Honest)", f"FP\n{metrics_ae['false_positives']:,}\n(False Alarm)"],
        [f"FN\n{metrics_ae['false_negatives']:,}\n(Missed Thief)", f"TP\n{metrics_ae['true_positives']:,}\n(Thief Caught!)"]
    ]
    sns.heatmap(cm_ae, annot=annot_matrix, fmt='', cmap='Greens', cbar=False,
                xticklabels=['Approved (Safe)', 'Alert (Fraud Triage)'],
                yticklabels=['Actual Normal', 'Actual Fraud'],
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    plt.title(f"Figure 10: Autoencoder Test Confusion Matrix (tau = {tau_ae:.3f})", fontsize=12, fontweight='bold')
    plt.xlabel('Model Prediction', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Truth', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/fig10_ae_test_confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved results/figures/fig10_ae_test_confusion_matrix.png")
    
    # Figure 11: Hybrid Test Confusion Matrix Heatmap
    cm_hyb = np.array([[metrics_hybrid['true_negatives'], metrics_hybrid['false_positives']],
                       [metrics_hybrid['false_negatives'], metrics_hybrid['true_positives']]])
    plt.figure(figsize=(7, 5.5))
    annot_matrix_hyb = [
        [f"TN\n{metrics_hybrid['true_negatives']:,}\n(Honest)", f"FP\n{metrics_hybrid['false_positives']:,}\n(False Alarm)"],
        [f"FN\n{metrics_hybrid['false_negatives']:,}\n(Missed Thief)", f"TP\n{metrics_hybrid['true_positives']:,}\n(Thief Caught!)"]
    ]
    sns.heatmap(cm_hyb, annot=annot_matrix_hyb, fmt='', cmap='Reds', cbar=False,
                xticklabels=['Approved (Safe)', 'Alert (Fraud Triage)'],
                yticklabels=['Actual Normal', 'Actual Fraud'],
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    plt.title(f"Figure 11: GMM+AE Hybrid Test Confusion Matrix (tau = {tau_hybrid:.3f})", fontsize=12, fontweight='bold')
    plt.xlabel('Model Prediction', fontsize=11, fontweight='bold')
    plt.ylabel('Actual Truth', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/fig11_hybrid_test_confusion_matrix.png', dpi=300)
    plt.close()
    print("Saved results/figures/fig11_hybrid_test_confusion_matrix.png")
    
    print("\nAll Phase 2 experimental benchmarks complete and validated successfully!")


if __name__ == '__main__':
    run_pipeline()
