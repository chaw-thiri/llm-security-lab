"""Classification metrics and validation-only threshold selection."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def metrics(labels, scores, threshold=0.5):
    y, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Need nonempty aligned labels and finite scores.")
    if not set(y).issubset({0, 1}):
        raise ValueError("Only binary labels are supported.")
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    both = len(set(y)) == 2
    return {"n": len(y), "threshold": float(threshold), "accuracy": float((pred == y).mean()),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "benign_fpr": float(fp / (fp + tn)) if fp + tn else None,
            "average_precision": float(average_precision_score(y, scores)) if both else None,
            "roc_auc": float(roc_auc_score(y, scores)) if both else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def select_threshold(labels, scores, max_fpr=0.02):
    """Maximize recall subject to an empirical benign-FPR budget, using validation only.

    Ties prefer lower FPR, then the higher threshold. A threshold just above 1
    intentionally permits a no-alert policy when the budget cannot otherwise be met.
    """
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0, 1]")
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(set(y)) != 2 or len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Threshold selection requires both classes and probabilities in [0, 1].")
    # Sort once; handle equal scores as one group so >= has identical semantics in training and inference.
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = fp = 0
    positives, negatives = int(y.sum()), int((y == 0).sum())
    best = (0.0, 0.0, float(np.nextafter(1.0, 2.0)))
    for i, label in enumerate(ys):
        tp += int(label == 1)
        fp += int(label == 0)
        if i + 1 < len(ps) and ps[i] == ps[i + 1]:
            continue
        fpr = fp / negatives
        candidate = (tp / positives, -fpr, float(ps[i]))
        if fpr <= max_fpr and candidate > best:
            best = candidate
    return best[2]
