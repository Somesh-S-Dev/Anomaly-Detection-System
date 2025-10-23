import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server/web environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
import shap
import numpy as np

def plot_roc_curve(y_true, scores):
    """Plot ROC curve with error handling."""
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")

def plot_pr_curve(y_true, scores):
    """Plot Precision-Recall curve with error handling."""
    try:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR curve', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Error plotting PR curve: {e}")

def plot_confusion_matrix(cm, labels=[0, 1]):
    """Plot confusion matrix with error handling."""
    try:
        plt.figure(figsize=(6, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format='d')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def plot_shap_summary(model, X, max_features=20):
    """Plot SHAP summary with error handling."""
    try:
        # Limit features for better visualization
        if X.shape[1] > max_features:
            print(f"Limiting SHAP analysis to top {max_features} features for better visualization")
            # Use feature importance or variance to select top features
            feature_importance = np.var(X, axis=0)
            top_features_idx = np.argsort(feature_importance)[-max_features:]
            X_subset = X[:, top_features_idx]
        else:
            X_subset = X
        
        explainer = shap.Explainer(model, X_subset)
        shap_values = explainer(X_subset)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_subset, show=False)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        print("SHAP analysis requires compatible model types (tree-based models work best)")

def plot_anomaly_scores(scores):
    """Plot anomaly score distribution with error handling."""
    try:
        plt.figure(figsize=(10, 6))
        
        # Handle infinite or extreme values
        scores_clean = scores.copy()
        scores_clean = np.where(np.isinf(scores_clean), np.nan, scores_clean)
        scores_clean = np.where(np.isnan(scores_clean), np.nanmedian(scores_clean), scores_clean)
        
        # Remove outliers for better visualization
        q1, q3 = np.percentile(scores_clean, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        scores_filtered = scores_clean[(scores_clean >= lower_bound) & (scores_clean <= upper_bound)]
        
        if len(scores_filtered) > 0:
            sns.histplot(scores_filtered, bins=50, kde=True)
            plt.title('Anomaly Score Distribution (Outliers Removed)')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
        else:
            # If too many outliers, use original scores
            sns.histplot(scores_clean, bins=50, kde=True)
            plt.title('Anomaly Score Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
        
        plt.show()
        
        # Print summary statistics
        print(f"Anomaly Score Statistics:")
        print(f"Mean: {np.mean(scores_clean):.4f}")
        print(f"Std: {np.std(scores_clean):.4f}")
        print(f"Min: {np.min(scores_clean):.4f}")
        print(f"Max: {np.max(scores_clean):.4f}")
        print(f"95th percentile: {np.percentile(scores_clean, 95):.4f}")
        
    except Exception as e:
        print(f"Error plotting anomaly scores: {e}")

def plot_feature_importance(X, feature_names=None):
    """Plot feature importance based on variance."""
    try:
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Calculate feature importance based on variance
        feature_importance = np.var(X, axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = min(20, len(feature_names))  # Show top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(top_features), feature_importance[sorted_idx][:top_features])
        plt.xticks(range(top_features), [feature_names[i] for i in sorted_idx[:top_features]], rotation=45, ha='right')
        plt.title('Feature Importance (Based on Variance)')
        plt.xlabel('Features')
        plt.ylabel('Variance')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting feature importance: {e}") 

def plot_missing_value_heatmap(df):
    """Plot a heatmap of missing values in the DataFrame."""
    try:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Value Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.tight_layout()
        # plt.show()  # Removed to prevent pop-up in web app
    except Exception as e:
        print(f"Error plotting missing value heatmap: {e}")

def plot_feature_histogram(df, col, save_path):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f'Histogram: {col}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_boxplot(df, col, save_path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col].dropna())
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_histograms(df, max_cols=5):
    """Plot histograms for all numerical features."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = min(max_cols, len(num_cols))
        n_rows = int(np.ceil(len(num_cols) / n_cols))
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(num_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f'Histogram: {col}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting feature histograms: {e}")

def plot_feature_boxplots(df, max_cols=5):
    """Plot boxplots for all numerical features."""
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = min(max_cols, len(num_cols))
        n_rows = int(np.ceil(len(num_cols) / n_cols))
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(num_cols):
            plt.subplot(n_rows, n_cols, i + 1)
            sns.boxplot(y=df[col].dropna())
            plt.title(f'Boxplot: {col}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting feature boxplots: {e}")

def plot_correlation_matrix(df):
    """Plot a correlation matrix heatmap for numerical features."""
    try:
        corr = df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        # plt.show()  # Removed to prevent pop-up in web app
    except Exception as e:
        print(f"Error plotting correlation matrix: {e}") 