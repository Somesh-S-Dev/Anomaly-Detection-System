import numpy as np
from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        self.model.fit(X, y)
    def predict(self, X, threshold=None):
        if hasattr(X, 'values'):
            X = X.values
        probs = self.model.predict_proba(X)[:, 1]
        
        # Clean probabilities - handle NaN and infinite values
        probs = np.where(np.isnan(probs), 0, probs)
        probs = np.where(np.isinf(probs), 0, probs)
        probs = np.clip(probs, 0, 1)  # Clip to [0, 1] range
        
        # Use dynamic threshold for anomaly detection
        if threshold is None:
            # Use 95th percentile as threshold for anomaly detection
            threshold = np.percentile(probs, 95)
        
        # Safe conversion to int
        preds = (probs > threshold)
        preds = np.where(np.isnan(preds), False, preds)
        preds = preds.astype(int)
        
        return preds
    def anomaly_score(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return self.model.predict_proba(X)[:, 1] 