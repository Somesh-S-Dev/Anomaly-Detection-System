from data_loader import DataLoader
from preprocessing import preprocess_data, create_anomaly_labels
from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from models.autoencoder import AutoencoderModel, DeepAutoencoderModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from evaluation import evaluate_model, show_top_n_anomalies
from visualization import plot_roc_curve, plot_pr_curve, plot_confusion_matrix, plot_shap_summary, plot_anomaly_scores
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import time
import random
import torch

# Global results storage (matching UI backend)
global_results = {
    'models': {},
    'evaluations': {},
    'anomalies': {},
    'preprocessing_info': {},
    'data_info': {}
}

def print_model_results(model_name, eval_results, training_time):
    """Print model results in the same format as UI backend."""
    print(f"\n{'='*60}")
    print(f"🔍 {model_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"{'-'*50}")
    print(f"Precision:  {eval_results['precision']:.4f} ({eval_results['precision']*100:.2f}%)")
    print(f"Recall:     {eval_results['recall']:.4f} ({eval_results['recall']*100:.2f}%)")
    print(f"F1-Score:   {eval_results['f1']:.4f} ({eval_results['f1']*100:.2f}%)")
    print(f"Accuracy:   {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
    print(f"ROC AUC:    {eval_results['roc_auc']:.4f} ({eval_results['roc_auc']*100:.2f}%)")
    print(f"PR AUC:     {eval_results['pr_auc']:.4f} ({eval_results['pr_auc']*100:.2f}%)")
    print(f"{'-'*50}")
    
    # Confusion Matrix
    cm = eval_results['confusion_matrix']
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"┌─────────────┬─────────────┐")
    print(f"│ Predicted   │ Predicted   │")
    print(f"│ Normal      │ Anomaly     │")
    print(f"├─────────────┼─────────────┤")
    print(f"│ Actual      │ {cm[0,0]:>9} │ {cm[0,1]:>9} │")
    print(f"│ Normal      │             │")
    print(f"├─────────────┼─────────────┤")
    print(f"│ Actual      │ {cm[1,0]:>9} │ {cm[1,1]:>9} │")
    print(f"│ Anomaly     │             │")
    print(f"└─────────────┴─────────────┘")

def print_comparison_summary():
    """Print comparison summary matching UI backend format."""
    if not global_results.get('evaluations'):
        print("\n❌ No evaluation results available")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Header
    print(f"{'Model':<18} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10} {'ROC AUC':<10}")
    print(f"{'-'*80}")
    
    # Results for each model
    for model_name, results in global_results['evaluations'].items():
        display_name = model_name.replace('_', ' ').title()
        print(f"{display_name:<18} {results['precision']:<10.4f} {results['recall']:<10.4f} "
              f"{results['f1']:<10.4f} {results['accuracy']:<10.4f} {results['roc_auc']:<10.4f}")
    
    print(f"{'-'*80}")
    
    # Find best models for each metric
    print(f"\n🏆 BEST PERFORMING MODELS:")
    best_precision = max(global_results['evaluations'].items(), key=lambda x: x[1]['precision'])
    best_recall = max(global_results['evaluations'].items(), key=lambda x: x[1]['recall'])
    best_f1 = max(global_results['evaluations'].items(), key=lambda x: x[1]['f1'])
    best_accuracy = max(global_results['evaluations'].items(), key=lambda x: x[1]['accuracy'])
    best_roc = max(global_results['evaluations'].items(), key=lambda x: x[1]['roc_auc'])
    
    print(f"   Best Precision:   {best_precision[0].replace('_', ' ').title()} ({best_precision[1]['precision']:.4f})")
    print(f"   Best Recall:      {best_recall[0].replace('_', ' ').title()} ({best_recall[1]['recall']:.4f})")
    print(f"   Best F1-Score:    {best_f1[0].replace('_', ' ').title()} ({best_f1[1]['f1']:.4f})")
    print(f"   Best Accuracy:    {best_accuracy[0].replace('_', ' ').title()} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"   Best ROC AUC:     {best_roc[0].replace('_', ' ').title()} ({best_roc[1]['roc_auc']:.4f})")

if __name__ == '__main__':
    print("🚀 FINANCIAL ANOMALY DETECTION SYSTEM v7.0")
    print("="*60)
    
    # Set random seeds for reproducible results (matching UI backend)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Load and explore data
    print("\n📂 Loading and exploring dataset...")
    loader = DataLoader()
    df = loader.load_data()
    
    # Show dataset information
    loader.show_class_imbalance()
    loader.show_basic_stats()
    loader.plot_distributions()

    # Preprocess data (matching UI backend approach)
    print("\n🔧 Preprocessing data...")
    X, y, preprocessing_info = preprocess_data(
        df, 
        target_column=loader.target_column,
        resampling_strategy='auto',  # Auto-select best strategy
        random_state=42
    )
    X = np.array(X)
    if y is not None:
        y = np.array(y)
    
    # Store preprocessing information
    global_results['preprocessing_info'] = preprocessing_info
    
    # Create synthetic labels if no target exists
    if y is None:
        print("\n⚠️  No target column found. Creating synthetic anomaly labels for evaluation...")
        y = create_anomaly_labels(X, contamination=0.01, random_state=42)
    
    # Split data for proper evaluation (70:30 split) - matching UI backend
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {X_train.shape[0]} (70%)")
    print(f"   Testing samples:  {X_test.shape[0]} (30%)")
    print(f"   Features:         {X_train.shape[1]}")
    
    # Handle target values - convert to integers and handle NaN values
    if y_test is not None:
        y_test = np.nan_to_num(y_test, nan=0.0).astype(int)
        print(f"   Normal samples:   {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
        print(f"   Anomaly samples:  {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")

    # Run Isolation Forest (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING ISOLATION FOREST")
    print(f"{'='*60}")
    print("Training Isolation Forest...")
    start_time = time.time()
    
    iso_model = IsolationForestModel(contamination=0.01, random_state=42)
    iso_model.fit(X_train)
    iso_preds = iso_model.predict(X_test)
    iso_scores = iso_model.anomaly_score(X_test)
    iso_eval = evaluate_model(y_test, iso_preds, iso_scores)
    training_time = time.time() - start_time
    
    print_model_results("Isolation Forest", iso_eval, training_time)
    
    # Store results (matching UI backend)
    global_results['models']['isolation_forest'] = iso_model
    global_results['evaluations']['isolation_forest'] = iso_eval
    global_results['anomalies']['isolation_forest'] = iso_scores

    # Run One-Class SVM (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING ONE-CLASS SVM")
    print(f"{'='*60}")
    print("Training One-Class SVM...")
    start_time = time.time()
    
    svm_model = OneClassSVMModel(nu=0.01, random_state=42)
    svm_model.fit(X_train)
    svm_preds = svm_model.predict(X_test)
    svm_scores = svm_model.anomaly_score(X_test)
    svm_eval = evaluate_model(y_test, svm_preds, svm_scores)
    training_time = time.time() - start_time
    
    print_model_results("One-Class SVM", svm_eval, training_time)
    
    # Store results
    global_results['models']['one_class_svm'] = svm_model
    global_results['evaluations']['one_class_svm'] = svm_eval
    global_results['anomalies']['one_class_svm'] = svm_scores

    # Run Autoencoder (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING AUTOENCODER")
    print(f"{'='*60}")
    print("Training Autoencoder...")
    start_time = time.time()
    
    auto_model = AutoencoderModel(input_dim=X_train.shape[1], random_state=42)
    auto_model.fit(X_train, epochs=10, verbose=0)
    auto_preds, auto_errors = auto_model.predict(X_test)
    auto_eval = evaluate_model(y_test, auto_preds, auto_errors)
    training_time = time.time() - start_time
    
    print_model_results("Autoencoder", auto_eval, training_time)
    
    # Store results
    global_results['models']['autoencoder'] = auto_model
    global_results['evaluations']['autoencoder'] = auto_eval
    global_results['anomalies']['autoencoder'] = auto_errors

    # Run Deep Autoencoder (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING DEEP AUTOENCODER")
    print(f"{'='*60}")
    print("Training Deep Autoencoder...")
    start_time = time.time()
    
    deep_auto_model = DeepAutoencoderModel(input_dim=X_train.shape[1])
    deep_auto_model.fit(X_train)
    deep_auto_errors = deep_auto_model.get_reconstruction_errors(X_test)
    deep_auto_preds = (deep_auto_errors > np.percentile(deep_auto_errors, 95)).astype(int)
    deep_auto_eval = evaluate_model(y_test, deep_auto_preds, deep_auto_errors)
    training_time = time.time() - start_time
    
    print_model_results("Deep Autoencoder", deep_auto_eval, training_time)
    
    # Store results
    global_results['models']['deep_autoencoder'] = deep_auto_model
    global_results['evaluations']['deep_autoencoder'] = deep_auto_eval
    global_results['anomalies']['deep_autoencoder'] = deep_auto_errors

    # Run XGBoost (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING XGBOOST")
    print(f"{'='*60}")
    print("Training XGBoost...")
    start_time = time.time()
    
    xgb_model = XGBoostModel(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)  # Uses dynamic threshold
    xgb_scores = xgb_model.anomaly_score(X_test)
    xgb_eval = evaluate_model(y_test, xgb_preds, xgb_scores)
    training_time = time.time() - start_time
    
    print_model_results("XGBoost", xgb_eval, training_time)
    
    # Store results
    global_results['models']['xgboost'] = xgb_model
    global_results['evaluations']['xgboost'] = xgb_eval
    global_results['anomalies']['xgboost'] = xgb_scores

    # Run Random Forest (matching UI backend)
    print(f"\n{'='*60}")
    print("🔍 RUNNING RANDOM FOREST")
    print(f"{'='*60}")
    print("Training Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)  # Uses dynamic threshold
    rf_scores = rf_model.anomaly_score(X_test)
    rf_eval = evaluate_model(y_test, rf_preds, rf_scores)
    training_time = time.time() - start_time
    
    print_model_results("Random Forest", rf_eval, training_time)
    
    # Store results
    global_results['models']['random_forest'] = rf_model
    global_results['evaluations']['random_forest'] = rf_eval
    global_results['anomalies']['random_forest'] = rf_scores

    # Print comparison summary
    print_comparison_summary()

    # Show top N anomalies
    print(f"\n{'='*60}")
    print(f"🔍 TOP ANOMALIES ANALYSIS")
    print(f"{'='*60}")
    print('\nTop 10 suspicious records (Isolation Forest):')
    try:
        top_anomalies = show_top_n_anomalies(df, iso_scores, n=10)
        print(top_anomalies)
    except Exception as e:
        print(f"Could not display top anomalies: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ ANOMALY DETECTION ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Successfully evaluated {len(global_results['evaluations'])} models")
    print(f"🎯 Best overall model: {max(global_results['evaluations'].items(), key=lambda x: x[1]['f1'])[0].replace('_', ' ').title()}")
    print(f"💡 Check the web interface at http://localhost:5000 for interactive visualizations") 