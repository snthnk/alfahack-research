import numpy as np

from fa_metrics import evaluate_metric


def test_rmse_is_root_of_mean_squared_error():
    assert evaluate_metric([0, 0], [3, 4], "rmse", "regression") == np.sqrt(12.5)


def test_macro_f1_uses_argmax_in_class_order():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    assert evaluate_metric(["a", "b", "b"], proba, "f1_macro", "binary", ["a", "b"]) == evaluate_metric(
        ["a", "b", "b"], np.array(["a", "b", "a"]), "f1_macro", "binary", ["a", "b"])


def test_unknown_classification_metric_falls_back_to_accuracy():
    proba = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert evaluate_metric(["a", "b"], proba, "something", "binary", ["a", "b"]) == 0.5
