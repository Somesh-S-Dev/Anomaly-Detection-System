from sklearn.svm import OneClassSVM
import numpy as np

class OneClassSVMModel:
    def __init__(self, nu=0.001, kernel='rbf', gamma='scale', random_state=42):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    def fit(self, X):
        if hasattr(X, 'values'):
            X = X.values
        self.model.fit(X)
    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        preds = self.model.predict(X)
        # OneClassSVM: -1=anomaly, 1=normal
        return np.where(preds == -1, 1, 0)
    def anomaly_score(self, X):
        if hasattr(X, 'values'):
            X = X.values
        scores = -self.model.decision_function(X)
        
        # Clean scores - handle NaN and infinite values
        scores = np.where(np.isnan(scores), 0, scores)
        scores = np.where(np.isinf(scores), 0, scores)
        
        return scores 