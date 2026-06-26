"""
Spec2Prop-Inorg: Traditional Baselines
======================================
Wrappers for sklearn baseline models (SVM, RF, LogReg, kNN, XGBoost).
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class BaselineModel:
    """Wrapper for scikit-learn models with optional PCA reduction."""
    
    def __init__(self, model_type: str, task_type: str = "classification", pca_dim: int = None, random_state: int = 42):
        self.model_type = model_type
        self.task_type = task_type
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=pca_dim, random_state=random_state) if pca_dim else None
        
        if task_type == "classification":
            if model_type == "SVM":
                self.model = SVC(kernel="rbf", probability=True, random_state=random_state)
            elif model_type == "RandomForest":
                self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
            elif model_type == "LogisticRegression":
                self.model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
            elif model_type == "kNN":
                self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            elif model_type == "XGBoost" and HAS_XGBOOST:
                self.model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=random_state)
            else:
                raise ValueError(f"Unknown or unavailable classification model: {model_type}")
        elif task_type == "regression":
            if model_type == "SVM":
                self.model = SVR(kernel="rbf")
            elif model_type == "RandomForest":
                self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
            elif model_type == "LogisticRegression":
                self.model = Ridge(random_state=random_state)
            elif model_type == "kNN":
                self.model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
            elif model_type == "XGBoost" and HAS_XGBOOST:
                self.model = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
            else:
                raise ValueError(f"Unknown or unavailable regression model: {model_type}")
        else:
            raise ValueError(f"Invalid task_type: {task_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.pca:
            X = self.pca.fit_transform(X)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray):
        if self.pca:
            X = self.pca.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only for classification")
        if self.pca:
            X = self.pca.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X)
            # Create dummy probabilities (1.0 for predicted class)
            n_classes = len(set(preds))
            probs = np.zeros((len(preds), n_classes))
            for i, p in enumerate(preds):
                probs[i, int(p)] = 1.0
            return probs
