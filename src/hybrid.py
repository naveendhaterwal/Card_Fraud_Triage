"""
Hybrid Score Fusion Module: GMM Density Scoring + Autoencoder Reconstruction Error
Team T047 | Phase 2 Deep Learning
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def normalize_scores(scores, method='rank'):
    """
    Normalize raw anomaly scores to a uniform [0, 1] scale.
    
    Args:
      scores: 1D np.ndarray of raw anomaly scores.
      method: 'rank' (empirical CDF / percentile rank in [0, 1])
              or 'minmax' (standard min-max normalization).
              
    Justification for 'rank':
      Raw GMM scores (-log p(x)) range from ~20 to ~2000+, while AE MSE scores
      range from ~0.05 to ~100+. Min-max is heavily distorted by extreme single-sample
      outliers. Rank normalization transforms both scores into their empirical percentile,
      preserving strict monotonic order while rendering their distributions directly comparable.
    """
    if method == 'rank':
        # rankdata gives ranks 1..N. Dividing by N yields values in (0, 1].
        return (rankdata(scores) - 1.0) / (len(scores) - 1.0)
    elif method == 'minmax':
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min == 0:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_hybrid_score(scores_gmm, scores_ae, alpha=0.5, norm_method='rank'):
    """
    Calculate fused anomaly score:
      S_hybrid = alpha * norm(S_gmm) + (1 - alpha) * norm(S_ae)
    """
    norm_gmm = normalize_scores(scores_gmm, method=norm_method)
    norm_ae = normalize_scores(scores_ae, method=norm_method)
    return alpha * norm_gmm + (1.0 - alpha) * norm_ae


def optimize_hybrid_alpha(val_gmm_scores, val_ae_scores, y_val, 
                          alphas=np.linspace(0.0, 1.0, 21), norm_method='rank'):
    """
    Find optimal blending coefficient alpha* strictly on the validation set.
    
    Returns:
      best_alpha: Float in [0, 1] maximizing validation Average Precision.
      best_ap: Float of highest validation AP.
      alpha_search_results: List of (alpha, ap, roc_auc) tuples.
    """
    results = []
    best_ap = -1.0
    best_alpha = 0.5
    
    norm_gmm = normalize_scores(val_gmm_scores, method=norm_method)
    norm_ae = normalize_scores(val_ae_scores, method=norm_method)
    
    for alpha in alphas:
        hybrid_scores = alpha * norm_gmm + (1.0 - alpha) * norm_ae
        ap = average_precision_score(y_val, hybrid_scores)
        roc = roc_auc_score(y_val, hybrid_scores)
        results.append((round(alpha, 3), ap, roc))
        
        if ap > best_ap:
            best_ap = ap
            best_alpha = round(alpha, 3)
            
    return best_alpha, best_ap, results
