from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
import numpy as np

def evaluate_model(y_true, y_pred, scores=None):
    """
    Evaluate model performance with robust error handling.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        scores: Anomaly scores (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    results = {}
    
    try:
        # Ensure we have both classes for evaluation
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_classes) < 2:
            print("Warning: Only one class found in predictions. Evaluation metrics may be unreliable.")
            # Add a dummy prediction to avoid errors
            y_pred_adj = y_pred.copy()
            if 0 not in y_pred_adj:
                y_pred_adj[0] = 0
            if 1 not in y_pred_adj:
                y_pred_adj[-1] = 1
        else:
            y_pred_adj = y_pred
        
        results['precision'] = precision_score(y_true, y_pred_adj, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred_adj, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred_adj, zero_division=0)
        results['accuracy'] = accuracy_score(y_true, y_pred_adj)
        
        # Use scores if available, otherwise use predictions
        score_values = scores if scores is not None else y_pred_adj
        
        # Handle ROC AUC
        try:
            results['roc_auc'] = roc_auc_score(y_true, score_values)
        except ValueError:
            results['roc_auc'] = 0.5  # Default for single class
        
        # Handle PR AUC
        try:
            results['pr_auc'] = average_precision_score(y_true, score_values)
        except ValueError:
            results['pr_auc'] = 0.0  # Default for single class
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred_adj)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        # Return default values
        results = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'roc_auc': 0.5,
            'pr_auc': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }
    
    return results

def show_top_n_anomalies(X, scores, n=10):
    """
    Show top N anomalies based on scores.
    
    Args:
        X: Original dataframe or feature matrix
        scores: Anomaly scores
        n: Number of top anomalies to show
    
    Returns:
        Top N anomalies
    """
    try:
        # Get indices of top anomalies
        idx = np.argsort(scores)[-n:][::-1]
        
        # Handle different input types
        if hasattr(X, 'iloc'):
            # X is a dataframe
            return X.iloc[idx]
        else:
            # X is a numpy array or similar
            return X[idx]
    
    except Exception as e:
        print(f"Error showing top anomalies: {e}")
        return f"Could not display top {n} anomalies due to error: {e}" 