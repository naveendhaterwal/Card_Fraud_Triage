"""
Comprehensive Technical Evaluation Module with Bootstrap Confidence Intervals
Team T047 | Card Fraud Triage
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)


def bootstrap_metric_ci(y_true, scores, metric_fn, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Compute non-parametric 95% bootstrap confidence intervals for any metric.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        # Ensure at least one positive and one negative sample in the bootstrap draw
        if len(np.unique(y_true[indices])) < 2:
            continue
        val = metric_fn(y_true[indices], scores[indices])
        bootstrapped_scores.append(val)
        
    bootstrapped_scores = np.array(bootstrapped_scores)
    lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def calibrate_threshold_at_fpr(val_scores, y_val, target_fpr=0.005):
    """
    Find threshold tau on validation set that yields an FPR <= target_fpr (e.g. 0.5%).
    FPR is calculated strictly on legitimate samples (y_val == 0).
    """
    legit_scores = val_scores[y_val == 0]
    # For anomaly scores where higher is more anomalous,
    # FPR is the fraction of normal samples with score >= tau.
    # To enforce FPR <= target_fpr, tau is the (1 - target_fpr) quantile.
    tau = np.percentile(legit_scores, 100 * (1.0 - target_fpr))
    return float(tau)


def evaluate_model_performance(y_test, test_scores, tau_calibrated, model_name="Autoencoder"):
    """
    Compute all primary metrics matching Phase 1 GMM benchmark.
    """
    test_ap = float(average_precision_score(y_test, test_scores))
    test_roc_auc = float(roc_auc_score(y_test, test_scores))
    
    # Bootstrap CIs
    ap_ci = bootstrap_metric_ci(y_test, test_scores, average_precision_score)
    roc_ci = bootstrap_metric_ci(y_test, test_scores, roc_auc_score)
    
    # Decisions at calibrated threshold
    y_pred = (test_scores >= tau_calibrated).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Recall bootstrap CI
    def recall_fn(y_t, sc):
        pred = (sc >= tau_calibrated).astype(int)
        cm_b = confusion_matrix(y_t, pred, labels=[0, 1])
        return cm_b[1, 1] / (cm_b[1, 1] + cm_b[1, 0]) if (cm_b[1, 1] + cm_b[1, 0]) > 0 else 0.0
        
    recall_ci = bootstrap_metric_ci(y_test, test_scores, recall_fn)
    
    # Triage Queue Precision at Top K
    sorted_indices = np.argsort(test_scores)[::-1]
    sorted_labels = y_test[sorted_indices]
    
    prec_top25 = float(np.mean(sorted_labels[:25]))
    prec_top50 = float(np.mean(sorted_labels[:50]))
    prec_top100 = float(np.mean(sorted_labels[:100]))
    
    def prec_top50_fn(y_t, sc):
        idx = np.argsort(sc)[::-1]
        return np.mean(y_t[idx][:50])
    prec_top50_ci = bootstrap_metric_ci(y_test, test_scores, prec_top50_fn)
    
    metrics = {
        'model': model_name,
        'test_average_precision': test_ap,
        'test_average_precision_95ci': [ap_ci[0], ap_ci[1]],
        'test_roc_auc': test_roc_auc,
        'test_roc_auc_95ci': [roc_ci[0], roc_ci[1]],
        'chosen_threshold': float(tau_calibrated),
        'recall_at_0.5pct_fpr': float(recall),
        'recall_at_0.5pct_fpr_95ci': [recall_ci[0], recall_ci[1]],
        'false_positive_rate': float(fpr),
        'precision_top_25': prec_top25,
        'precision_top_50': prec_top50,
        'precision_top_50_95ci': [prec_top50_ci[0], prec_top50_ci[1]],
        'precision_top_100': prec_top100,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }
    return metrics
