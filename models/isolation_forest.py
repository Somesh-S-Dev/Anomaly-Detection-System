# isolation_forest.py
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestModel:
    def __init__(self, contamination='auto', random_state=42, n_estimators=100, max_samples='auto', max_features=1.0):
        self.model = IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features
        )
    
    def fit(self, X):
        if hasattr(X, 'values'):
            X = X.values
        self.model.fit(X)
    
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)
    
    def anomaly_score(self, X):
        if hasattr(X, 'values'):
            X = X.values
        return -self.model.decision_function(X)
