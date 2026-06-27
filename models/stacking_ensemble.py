"""
Spec2Prop-Inorg: Stacking / Probability Ensemble
==================================================
Combines predictions from multiple classifiers (LightGBM, XGBoost,
TabPFN, CatBoost) via probability averaging or learned stacking.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


class ProbabilityEnsemble:
    """
    Ensemble by weighted averaging of predicted probabilities.

    This is the simplest and most robust ensemble method. Each model
    contributes its class probability estimates, which are averaged
    (optionally with weights) to produce the final prediction.

    Typically adds 1–3% over the best single model.
    """

    def __init__(self, weights=None):
        """
        Parameters
        ----------
        weights : list of float or None
            Per-model weights for averaging. If None, uses equal weights.
        """
        self.weights = weights

    def predict_proba(self, proba_list):
        """
        Average probabilities from multiple models.

        Parameters
        ----------
        proba_list : list of np.ndarray
            Each array has shape (N, num_classes).

        Returns
        -------
        np.ndarray
            Averaged probabilities, shape (N, num_classes).
        """
        if self.weights is None:
            w = [1.0 / len(proba_list)] * len(proba_list)
        else:
            w = self.weights
            total = sum(w)
            w = [wi / total for wi in w]

        result = np.zeros_like(proba_list[0])
        for wi, pi in zip(w, proba_list):
            result += wi * pi
        return result

    def predict(self, proba_list):
        """Return class with highest averaged probability."""
        avg_proba = self.predict_proba(proba_list)
        return np.argmax(avg_proba, axis=1)


class StackedEnsemble:
    """
    Stacked generalization (Wolpert, 1992).

    Level-0: Diverse base model probability predictions.
    Level-1: Logistic Regression meta-learner trained on concatenated
             probability vectors.

    Use out-of-fold predictions for training to prevent data leakage.
    """

    def __init__(self, meta_learner=None):
        """
        Parameters
        ----------
        meta_learner : sklearn estimator or None
            If None, uses LogisticRegression with balanced class weights.
        """
        if meta_learner is None:
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )
        else:
            self.meta_learner = meta_learner

    def fit(self, proba_list_train, y_train):
        """
        Train the meta-learner on concatenated base model probabilities.

        Parameters
        ----------
        proba_list_train : list of np.ndarray
            Each array has shape (N_train, num_classes).
        y_train : np.ndarray
            True labels, shape (N_train,).
        """
        meta_features = np.hstack(proba_list_train)
        self.meta_learner.fit(meta_features, y_train)

    def predict_proba(self, proba_list_test):
        """Predict using the meta-learner."""
        meta_features = np.hstack(proba_list_test)
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, proba_list_test):
        """Predict class labels using the meta-learner."""
        meta_features = np.hstack(proba_list_test)
        return self.meta_learner.predict(meta_features)


def evaluate_ensemble(y_true, proba_list, model_names, method="average", weights=None):
    """
    Convenience function: evaluate an ensemble of models.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    proba_list : list of np.ndarray
        Per-model probability predictions.
    model_names : list of str
        Human-readable model names.
    method : str
        "average" or "stacking".
    weights : list of float or None
        Weights for probability averaging.

    Returns
    -------
    dict
        Results including per-model and ensemble metrics.
    """
    results = {}

    # Per-model results
    for name, proba in zip(model_names, proba_list):
        preds = np.argmax(proba, axis=1)
        results[name] = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "macro_f1": float(f1_score(y_true, preds, average="macro")),
        }

    # Ensemble
    if method == "average":
        ensemble = ProbabilityEnsemble(weights=weights)
        preds = ensemble.predict(proba_list)
        avg_proba = ensemble.predict_proba(proba_list)
    else:
        raise ValueError(f"Method '{method}' requires fit(). Use ProbabilityEnsemble or StackedEnsemble directly.")

    results["Ensemble"] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "method": method,
    }

    return results
