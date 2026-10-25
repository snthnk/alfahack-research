"""Метрики для проверки пайплайна: классификация получает матрицу вероятностей в порядке classes."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def evaluate_metric(y_true, y_pred, metric: str, task_type: str, classes: list[str] | None = None) -> float:
    m = metric.lower().replace("-", "_")
    if task_type == "regression":
        t, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float).ravel()
        if m == "mse":
            return float(mean_squared_error(t, p))
        if m == "mae":
            return float(mean_absolute_error(t, p))
        if m == "r2":
            return float(r2_score(t, p))
        if m == "rmsle":
            return float(np.sqrt(mean_squared_error(np.log1p(np.clip(t, 0, None)), np.log1p(np.clip(p, 0, None)))))
        return float(np.sqrt(mean_squared_error(t, p)))

    names = [str(c) for c in classes or []]
    truth = np.array([str(v) for v in np.asarray(y_true)])
    proba = np.asarray(y_pred)
    labels = np.array(names)[proba.argmax(axis=1)] if proba.ndim == 2 and names else proba
    if m in ("roc_auc", "auc"):
        if task_type == "binary":
            return float(roc_auc_score(truth == names[1], proba[:, 1]))
        return float(roc_auc_score(truth, proba, labels=names, multi_class="ovr"))
    if m == "log_loss":
        return float(log_loss(truth, proba, labels=names))
    if m in ("f1_macro", "macro_f1"):
        return float(f1_score(truth, labels, average="macro", zero_division=0))
    if m in ("f1_weighted", "weighted_f1"):
        return float(f1_score(truth, labels, average="weighted", zero_division=0))
    if m == "f1" and task_type == "binary" and len(names) > 1:
        return float(f1_score(truth, labels, pos_label=names[1], zero_division=0))
    if m == "f1":
        return float(f1_score(truth, labels, average="macro", zero_division=0))
    return float(accuracy_score(truth, labels))
