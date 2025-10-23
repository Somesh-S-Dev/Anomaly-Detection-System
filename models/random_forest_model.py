import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        self.model.fit(X, y)
    def predict(self, X, threshold=None):
        if hasattr(X, 'values'):
            X = X.values
        probs = np.array(self.model.predict_proba(X))
        if probs.ndim == 2 and probs.shape[1] == 2:
            anomaly_probs = probs[:, 1]
        else:
            anomaly_probs = probs.ravel()
        
        # Clean probabilities - handle NaN and infinite values
        anomaly_probs = np.where(np.isnan(anomaly_probs), 0, anomaly_probs)
        anomaly_probs = np.where(np.isinf(anomaly_probs), 0, anomaly_probs)
        anomaly_probs = np.clip(anomaly_probs, 0, 1)  # Clip to [0, 1] range
        
        # Use dynamic threshold for anomaly detection
        if threshold is None:
            # Use 95th percentile as threshold for anomaly detection
            threshold = np.percentile(anomaly_probs, 95)
        
        # Safe conversion to int
        preds = (anomaly_probs > threshold)
        preds = np.where(np.isnan(preds), False, preds)
        preds = preds.astype(int)
        
        return preds
    def anomaly_score(self, X):
        if hasattr(X, 'values'):
            X = X.values
        probs = np.array(self.model.predict_proba(X))
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        else:
            return probs.ravel() 