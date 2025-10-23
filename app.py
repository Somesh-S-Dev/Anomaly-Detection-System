from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime
import os
import pickle
import joblib

import matplotlib.pyplot as plt

# Import our modules
from data_loader import DataLoader
from preprocessing import preprocess_data, create_anomaly_labels
from models.isolation_forest import IsolationForestModel
from models.one_class_svm import OneClassSVMModel
from models.autoencoder import AutoencoderModel, DeepAutoencoderModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from evaluation import evaluate_model, show_top_n_anomalies
from visualization import plot_roc_curve, plot_pr_curve, plot_confusion_matrix, plot_shap_summary, plot_anomaly_scores
from research_analysis import LiteratureReview, ModelComparison, Recommendations
from hyperparameter_tuning import run_hyperparameter_tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
from eda_utils import run_eda_pipeline

def evaluate_tuned_models(X, y, tuning_results, random_state=42):
    """Evaluate tuned models and return full metrics."""
    from models.isolation_forest import IsolationForestModel
    from models.one_class_svm import OneClassSVMModel
    from models.autoencoder import AutoencoderModel, DeepAutoencoderModel
    from models.random_forest_model import RandomForestModel
    from models.xgboost_model import XGBoostModel
    
    print(f"🔍 Evaluating {len(tuning_results)} tuned models...")
    print(f"🔍 Tuning results keys: {list(tuning_results.keys())}")
    
    # Split data for evaluation (70:30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    evaluations = {}
    
    # Evaluate each tuned model
    for model_name, tuning_result in tuning_results.items():
        try:
            print(f"🔍 Evaluating tuned {model_name}...")
            print(f"  Parameters: {tuning_result['params']}")
            
            if model_name == 'isolation_forest':
                # Isolation Forest parameters - filter out unsupported parameters
                params = tuning_result['params'].copy()
                # Remove parameters that don't exist in our model
                params.pop('max_features', None)
                params.pop('max_samples', None)
                params.pop('n_estimators', None)
                
                model = IsolationForestModel(**params)
                model.fit(X_train)
                y_pred = model.predict(X_test)
                y_scores = model.anomaly_score(X_test)
                
            elif model_name == 'one_class_svm':
                # One-Class SVM parameters go to constructor
                model = OneClassSVMModel(**tuning_result['params'])
                model.fit(X_train)
                y_pred = model.predict(X_test)
                y_scores = model.anomaly_score(X_test)
                
            elif model_name == 'autoencoder':
                # Autoencoder parameters - filter out unsupported parameters
                auto_params = tuning_result['params'].copy()
                epochs = auto_params.pop('epochs', 10)
                batch_size = auto_params.pop('batch_size', 256)
                # Remove parameters that don't exist in our model constructor
                auto_params.pop('learning_rate', None)
                
                model = AutoencoderModel(input_dim=X_train.shape[1], **auto_params)
                model.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=0)
                y_pred, y_scores = model.predict(X_test)
                
            elif model_name == 'deep_autoencoder':
                # Deep Autoencoder parameters go to constructor
                model = DeepAutoencoderModel(input_dim=X_train.shape[1], **tuning_result['params'])
                model.fit(X_train)
                y_pred, y_scores = model.predict(X_test)
                
            elif model_name == 'random_forest':
                # Random Forest parameters go to constructor
                model = RandomForestModel(**tuning_result['params'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)  # Uses dynamic threshold
                y_scores = model.anomaly_score(X_test)
                
            elif model_name == 'xgboost':
                # XGBoost parameters go to constructor
                model = XGBoostModel(**tuning_result['params'])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)  # Uses dynamic threshold
                y_scores = model.anomaly_score(X_test)
                
            else:
                continue
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_scores) if len(np.unique(y_scores)) > 1 else 0.5
            pr_auc = average_precision_score(y_test, y_scores)
            
            # Debug information
            print(f"  Debug - Predictions: {np.sum(y_pred)}/{len(y_pred)} ({np.mean(y_pred):.3f})")
            print(f"  Debug - True labels: {np.sum(y_test)}/{len(y_test)} ({np.mean(y_test):.3f})")
            print(f"  Debug - Scores range: {np.min(y_scores):.3f} to {np.max(y_scores):.3f}")
            if model_name in ['random_forest', 'xgboost']:
                # Show threshold information for supervised models
                threshold = np.percentile(y_scores, 95)
                print(f"  Debug - Dynamic threshold (95th percentile): {threshold:.3f}")
                print(f"  Debug - Scores above threshold: {np.sum(y_scores > threshold)}")
            
            evaluations[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
            
            print(f"✅ {model_name} evaluation complete")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            evaluations[model_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0,
                'roc_auc': 0.5,
                'pr_auc': 0.0
            }
    
    print(f"🔍 Final evaluations: {list(evaluations.keys())}")
    print(f"🔍 Evaluation sample: {next(iter(evaluations.values())) if evaluations else 'No evaluations'}")
    return evaluations

app = Flask(__name__)
app.config['SECRET_KEY'] = 'anomaly_detection_secret_key'


# Global variables to store results
global_results = {
    'data_info': None,
    'models': {},
    'evaluations': {},
    'anomalies': {},
    'tuning_results': {},
    'literature': None,
    'recommendations': None,
    'comparison': None
}

def retrain_models_for_prediction():
    """Re-train models for prediction when they're not available in memory."""
    print("🔄 Re-training models for prediction...")
    print(f"🔍 Current evaluations: {list(global_results.get('evaluations', {}).keys())}")
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    
    # Preprocess data with imbalance handling
    X, y, preprocessing_info = preprocess_data(
        df, 
        target_column=loader.target_column,
        resampling_strategy='auto',  # Auto-select best strategy
        random_state=42
    )
    X = np.array(X)
    if y is not None:
        y = np.array(y)
    
    # Create synthetic labels if no target exists
    if y is None:
        y = create_anomaly_labels(X, contamination=0.01, random_state=42)
    
    # Split data for training (use full dataset for prediction models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Re-train all models
    print("Training Isolation Forest...")
    iso_model = IsolationForestModel(contamination=0.01, random_state=42)
    iso_model.fit(X_train)
    
    print("Training One-Class SVM...")
    svm_model = OneClassSVMModel(nu=0.01, random_state=42)
    svm_model.fit(X_train)
    
    print("Training Autoencoder...")
    auto_model = AutoencoderModel(input_dim=X_train.shape[1], random_state=42)
    auto_model.fit(X_train, epochs=10, verbose=0)
    
    print("Training Deep Autoencoder...")
    deep_auto_model = DeepAutoencoderModel(input_dim=X_train.shape[1])
    deep_auto_model.fit(X_train)
    
    print("Training XGBoost...")
    xgb_model = XGBoostModel(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    print("Training Random Forest...")
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train)
    
    # Store models
    global_results['models'] = {
        'isolation_forest': iso_model,
        'one_class_svm': svm_model,
        'autoencoder': auto_model,
        'deep_autoencoder': deep_auto_model,
        'xgboost': xgb_model,
        'random_forest': rf_model
    }
    
    print(f"✅ Models re-trained successfully for prediction")
    print(f"🔍 Stored models: {list(global_results['models'].keys())}")
    print(f"🔍 Models count after retrain: {len(global_results['models'])}")

def save_results_to_file():
    """Save results to a temporary file for persistence."""
    import pickle
    import tempfile
    import os
    import joblib
    
    # Create a temporary file to store results
    temp_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_results.pkl')
    models_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_models.pkl')
    
    print(f"💾 Saving results to {temp_file}")
    print(f"💾 Current global_results keys: {list(global_results.keys())}")
    print(f"💾 Evaluations available: {bool(global_results.get('evaluations'))}")
    print(f"💾 Recommendations available: {bool(global_results.get('recommendations'))}")
    
    # Convert numpy arrays to lists for serialization
    serializable_results = {}
    for key, value in global_results.items():
        if key == 'models':
            # Save models separately using joblib for better compatibility
            try:
                joblib.dump(value, models_file)
                print(f"Models saved to {models_file}")
                # Keep a reference that models were saved
                serializable_results[key] = {'saved': True}
            except Exception as e:
                print(f"Warning: Could not save models: {e}")
                serializable_results[key] = {'saved': False}
        elif key == 'anomalies':
            # Convert numpy arrays to lists
            serializable_results[key] = {}
            for model, scores in value.items():
                if isinstance(scores, np.ndarray):
                    serializable_results[key][model] = scores.tolist()
                else:
                    serializable_results[key][model] = scores
        elif key == 'evaluations':
            # Convert evaluation results to be JSON serializable
            serializable_results[key] = {}
            for model_name, eval_results in value.items():
                serializable_results[key][model_name] = {}
                for eval_key, eval_value in eval_results.items():
                    if isinstance(eval_value, np.ndarray):
                        serializable_results[key][model_name][eval_key] = eval_value.tolist()
                    elif isinstance(eval_value, (np.integer, np.floating)):
                        serializable_results[key][model_name][eval_key] = float(eval_value)
                    else:
                        serializable_results[key][model_name][eval_key] = eval_value
        elif key == 'tuning_results':
            # Convert tuning results to be JSON serializable
            serializable_results[key] = {}
            for model_name, eval_results in value.items():
                serializable_results[key][model_name] = {}
                for eval_key, eval_value in eval_results.items():
                    if isinstance(eval_value, np.ndarray):
                        serializable_results[key][model_name][eval_key] = eval_value.tolist()
                    elif isinstance(eval_value, (np.integer, np.floating)):
                        serializable_results[key][model_name][eval_key] = float(eval_value)
                    else:
                        serializable_results[key][model_name][eval_key] = eval_value
        elif key in ['recommendations', 'comparison']:
            # Ensure recommendations and comparison data are properly serialized
            serializable_results[key] = value
        else:
            serializable_results[key] = value
    
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(serializable_results, f)
        print(f"✅ Results saved to {temp_file}")
        print(f"💾 Saved keys: {list(serializable_results.keys())}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def load_results_from_file():
    """Load results from temporary file."""
    import pickle
    import tempfile
    import os
    import numpy as np
    import joblib
    
    temp_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_results.pkl')
    models_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_models.pkl')
    
    print(f"📂 Checking for saved results at {temp_file}")
    print(f"📂 File exists: {os.path.exists(temp_file)}")
    
    if os.path.exists(temp_file):
        try:
            with open(temp_file, 'rb') as f:
                loaded_results = pickle.load(f)
            
            # Try to load models if they were saved
            if 'models' in loaded_results and loaded_results['models'].get('saved', False):
                try:
                    if os.path.exists(models_file):
                        models = joblib.load(models_file)
                        loaded_results['models'] = models
                        print(f"Models loaded from {models_file}")
                    else:
                        print("Models file not found, models will need to be retrained")
                        loaded_results['models'] = {}
                except Exception as e:
                    print(f"Error loading models: {e}")
                    loaded_results['models'] = {}
            else:
                loaded_results['models'] = {}
            
            # Convert lists back to numpy arrays for anomalies
            if 'anomalies' in loaded_results:
                for model, scores in loaded_results['anomalies'].items():
                    if isinstance(scores, list):
                        loaded_results['anomalies'][model] = np.array(scores)
            
            # Convert evaluation results back to proper format
            if 'evaluations' in loaded_results:
                for model_name, eval_results in loaded_results['evaluations'].items():
                    for eval_key, eval_value in eval_results.items():
                        if isinstance(eval_value, list) and eval_key == 'confusion_matrix':
                            loaded_results['evaluations'][model_name][eval_key] = np.array(eval_value)
            
            # Convert tuning results back to proper format
            if 'tuning_results' in loaded_results:
                for model_name, eval_results in loaded_results['tuning_results'].items():
                    for eval_key, eval_value in eval_results.items():
                        if isinstance(eval_value, list) and eval_key == 'confusion_matrix':
                            loaded_results['tuning_results'][model_name][eval_key] = np.array(eval_value)
            
            global_results.update(loaded_results)
            print(f"✅ Results loaded from {temp_file}")
            print(f"📂 Loaded keys: {list(loaded_results.keys())}")
            print(f"📂 Evaluations loaded: {bool(loaded_results.get('evaluations'))}")
            print(f"📂 Recommendations loaded: {bool(loaded_results.get('recommendations'))}")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
    
    print("📂 No saved results file found")
    return False

@app.after_request
def add_header(response):
    """Add headers to prevent caching."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    """Main dashboard page."""
    # Load results from file to ensure data persistence across page refreshes
    load_results_from_file()
    return render_template('index.html')

@app.route('/literature')
def literature():
    """Literature review page."""
    if global_results['literature'] is None:
        literature_review = LiteratureReview()
        global_results['literature'] = literature_review.get_literature_summary()
    
    return render_template('literature.html', literature=global_results['literature'])

@app.route('/data_analysis')
def data_analysis():
    csv_path = "SSBCI-Transactions-Dataset.csv"  # Update path as needed
    _, summary = run_eda_pipeline(csv_path)
    return render_template('data_analysis.html', summary=summary)

@app.route('/eda_plots')
def eda_plots():
    """Dedicated EDA plots visualization page."""
    # Ensure EDA plots are generated and available
    if global_results.get('eda_plots') is None or global_results.get('data_info') is None:
        # Trigger data_analysis logic to generate plots if not present
        from flask import redirect, url_for
        return redirect(url_for('data_analysis'))
    return render_template('eda_plots.html', data_info=global_results['data_info'], eda_plots=global_results['eda_plots'])

@app.route('/run_models', methods=['POST'])
def run_models():
    """Run all anomaly detection models."""
    try:
        # Set random seeds for reproducible results
        import random
        import numpy as np
        import torch
        
        # Set seeds for all random components
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Load and preprocess data
        loader = DataLoader()
        df = loader.load_data()
        
        # Preprocess data with imbalance handling
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
            y = create_anomaly_labels(X, contamination=0.01, random_state=42)
        
        # Split data for proper evaluation (70:30 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training on {X_train.shape[0]} samples (70%), testing on {X_test.shape[0]} samples (30%)")
        
        # Run Isolation Forest
        print("Running Isolation Forest...")
        iso_model = IsolationForestModel(contamination=0.01, random_state=42)
        iso_model.fit(X_train)
        iso_preds = iso_model.predict(X_test)
        iso_scores = iso_model.anomaly_score(X_test)
        iso_eval = evaluate_model(y_test, iso_preds, iso_scores)
        
        # Run One-Class SVM
        print("Running One-Class SVM...")
        svm_model = OneClassSVMModel(nu=0.01, random_state=42)
        svm_model.fit(X_train)
        svm_preds = svm_model.predict(X_test)
        svm_scores = svm_model.anomaly_score(X_test)
        svm_eval = evaluate_model(y_test, svm_preds, svm_scores)
        
        # Run Autoencoder
        print("Running Autoencoder...")
        auto_model = AutoencoderModel(input_dim=X_train.shape[1], random_state=42)
        auto_model.fit(X_train, epochs=10, verbose=0)
        auto_preds, auto_errors = auto_model.predict(X_test)
        auto_eval = evaluate_model(y_test, auto_preds, auto_errors)
        
        # Run Deep Autoencoder
        print("Running Deep Autoencoder...")
        deep_auto_model = DeepAutoencoderModel(input_dim=X_train.shape[1])
        deep_auto_model.fit(X_train)
        deep_auto_errors = deep_auto_model.get_reconstruction_errors(X_test)
        deep_auto_preds = (deep_auto_errors > np.percentile(deep_auto_errors, 95)).astype(int)
        deep_auto_eval = evaluate_model(y_test, deep_auto_preds, deep_auto_errors)

        # Run XGBoost
        print("Running XGBoost...")
        xgb_model = XGBoostModel(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)  # Uses dynamic threshold
        xgb_scores = xgb_model.anomaly_score(X_test)
        xgb_eval = evaluate_model(y_test, xgb_preds, xgb_scores)

        # Run Random Forest
        print("Running Random Forest...")
        rf_model = RandomForestModel()
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)  # Uses dynamic threshold
        rf_scores = rf_model.anomaly_score(X_test)
        rf_eval = evaluate_model(y_test, rf_preds, rf_scores)

        # Store results
        global_results['models'] = {
            'isolation_forest': iso_model,
            'one_class_svm': svm_model,
            'autoencoder': auto_model,
            'deep_autoencoder': deep_auto_model,
            'xgboost': xgb_model,
            'random_forest': rf_model
        }
        global_results['evaluations'] = {
            'isolation_forest': iso_eval,
            'one_class_svm': svm_eval,
            'autoencoder': auto_eval,
            'deep_autoencoder': deep_auto_eval,
            'xgboost': xgb_eval,
            'random_forest': rf_eval
        }
        global_results['anomalies'] = {
            'isolation_forest': iso_scores,
            'one_class_svm': svm_scores,
            'autoencoder': auto_errors,
            'deep_autoencoder': deep_auto_errors,
            'xgboost': xgb_scores,
            'random_forest': rf_scores
        }
        
        # Save results to file for persistence
        save_results_to_file()
        
        # Generate comparison analysis
        comparison = ModelComparison()
        comparison_results = comparison.compare_models(
            global_results['evaluations'],
            global_results['anomalies']
        )
        
        # Generate recommendations
        recommendations = Recommendations()
        recs = recommendations.generate_recommendations(
            global_results['evaluations'],
            global_results['data_info']
        )
        
        # Store comparison and recommendations in global_results for persistence
        global_results['comparison'] = comparison_results
        global_results['recommendations'] = recs
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'comparison': comparison_results,
            'recommendations': recs
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error running models: {str(e)}'
        })

@app.route('/tune_hyperparameters', methods=['POST'])
def tune_hyperparameters():
    """Run hyperparameter tuning for all models."""
    try:
        # Set random seeds for reproducible results
        import random
        import numpy as np
        import torch
        
        # Set seeds for all random components
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Load and preprocess data
        loader = DataLoader()
        df = loader.load_data()
        
        # Preprocess data with imbalance handling
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
            y = create_anomaly_labels(X, contamination=0.01, random_state=42)
        
        # Run hyperparameter tuning
        print("🚀 Starting hyperparameter tuning...")
        tuning_results = run_hyperparameter_tuning(X, y, random_state=42)
        
        # Evaluate tuned models to get full metrics
        print("📊 Evaluating tuned models...")
        tuned_evaluations = evaluate_tuned_models(X, y, tuning_results['results'], random_state=42)
        
        # Store both the original tuning results and the full evaluations
        global_results['tuning_results'] = tuned_evaluations
        global_results['tuning_params'] = tuning_results['results']  # Store the best parameters too
        
        # Generate comparison analysis for tuned models
        comparison = ModelComparison()
        comparison_results = comparison.compare_models(
            tuned_evaluations,
            global_results.get('anomalies', {})
        )
        
        # Generate recommendations for tuned models
        recommendations = Recommendations()
        recs = recommendations.generate_recommendations(
            tuned_evaluations,
            global_results.get('data_info', {})
        )
        
        # Update comparison and recommendations with tuned model results
        global_results['comparison'] = comparison_results
        global_results['recommendations'] = recs
        
        # Save results to file for persistence
        save_results_to_file()
        
        return jsonify({
            'status': 'success',
            'message': 'Hyperparameter tuning completed successfully',
            'results': tuning_results,
            'comparison': comparison_results,
            'recommendations': recs
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error in hyperparameter tuning: {str(e)}'
        })

@app.route('/model_comparison')
def model_comparison():
    load_results_from_file()  # ensure everything is loaded

    if not global_results.get('evaluations'):
        return render_template('model_comparison.html', has_data=False)

    models = list(global_results['evaluations'].keys())
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'pr_auc']

    # Combine default and tuned metrics for side-by-side display
    comparison_data = {}
    for metric in metrics:
        comparison_data[metric] = []
        for model in models:
            default_val = global_results['evaluations'][model].get(metric, None)
            # Safely get tuning results, defaulting to empty dict if not available
            tuning_results = global_results.get('tuning_results', {})
            tuned_val = tuning_results.get(model, {}).get(metric, None)
            
            comparison_data[metric].append({
                "model": model,
                "default": default_val,
                "tuned": tuned_val
            })

    # Prepare default and tuned results for template
    default_results = global_results.get('evaluations', {})
    tuned_results = global_results.get('tuning_results', {})
    
    return render_template(
        'model_comparison.html',
        models=models,
        metrics=metrics,
        comparison_data=comparison_data,
        default_results=default_results,
        tuned_results=tuned_results,
        has_data=True
    )

@app.route('/api/model_results')
def api_model_results():
    """Return model evaluation results for dashboard charts."""
    result_type = request.args.get('type', 'default')
    
    # Try to load results from file if not in memory
    if not global_results.get('evaluations'):
        load_results_from_file()
    
    if result_type == 'hyperparam':
        results = global_results.get('tuning_results', {})
    else:
        results = global_results.get('evaluations', {})
    
    if not results:
        return jsonify({'error': 'No results available'})
    
    # Convert numpy arrays and numpy types to serializable types
    serializable_results = {}
    for model_name, eval_results in results.items():
        serializable_results[model_name] = {}
        for key, value in eval_results.items():
            if hasattr(value, 'tolist'):
                serializable_results[model_name][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[model_name][key] = float(value)
            else:
                serializable_results[model_name][key] = value
    return jsonify(serializable_results)

@app.route('/api/anomaly_scores')
def api_anomaly_scores():
    """API endpoint for anomaly scores."""
    # Try to load results from file if not in memory
    if not global_results.get('anomalies'):
        load_results_from_file()
    
    if not global_results['anomalies']:
        return jsonify({'error': 'No anomaly scores available', 'message': 'Please run models first'})
    
    # Return normalized sample of scores for visualization
    sample_scores = {}
    for model, scores in global_results['anomalies'].items():
        # Convert numpy array to list and take first 1000 scores
        if isinstance(scores, np.ndarray):
            scores_array = scores[:1000]
        else:
            scores_array = np.array(scores[:1000])
        
        # Normalize scores to a reasonable range (0-100)
        if len(scores_array) > 0:
            # Remove infinite values
            scores_array = np.where(np.isinf(scores_array), np.nan, scores_array)
            
            # Handle extreme values by using percentile-based normalization
            valid_scores = scores_array[~np.isnan(scores_array)]
            if len(valid_scores) > 0:
                # Use 95th percentile as upper bound to handle outliers
                upper_bound = np.percentile(valid_scores, 95)
                lower_bound = np.min(valid_scores)
                
                # Normalize to 0-100 range
                if upper_bound > lower_bound:
                    normalized_scores = np.clip(
                        (scores_array - lower_bound) / (upper_bound - lower_bound) * 100, 
                        0, 100
                    )
                else:
                    normalized_scores = np.zeros_like(scores_array)
                
                # Replace NaN with 0
                normalized_scores = np.where(np.isnan(normalized_scores), 0, normalized_scores)
                sample_scores[model] = normalized_scores.tolist()
            else:
                sample_scores[model] = [0] * len(scores_array)
        else:
            sample_scores[model] = []
    
    return jsonify(sample_scores)

@app.route('/api/anomaly_scores_raw')
def api_anomaly_scores_raw():
    """API endpoint for raw (unnormalized) anomaly scores for debugging."""
    # Try to load results from file if not in memory
    if not global_results.get('anomalies'):
        load_results_from_file()
    
    if not global_results['anomalies']:
        return jsonify({'error': 'No anomaly scores available', 'message': 'Please run models first'})
    
    # Return raw scores for debugging
    raw_scores = {}
    for model, scores in global_results['anomalies'].items():
        # Convert numpy array to list and take first 1000 scores
        if isinstance(scores, np.ndarray):
            scores_array = scores[:1000]
        else:
            scores_array = np.array(scores[:1000])
        
        # Get statistics for debugging
        if len(scores_array) > 0:
            valid_scores = scores_array[~np.isnan(scores_array)]
            if len(valid_scores) > 0:
                stats = {
                    'min': float(np.min(valid_scores)),
                    'max': float(np.max(valid_scores)),
                    'mean': float(np.mean(valid_scores)),
                    'std': float(np.std(valid_scores)),
                    'percentile_95': float(np.percentile(valid_scores, 95)),
                    'percentile_99': float(np.percentile(valid_scores, 99)),
                    'has_infinite': bool(np.any(np.isinf(valid_scores))),
                    'has_nan': bool(np.any(np.isnan(valid_scores)))
                }
                raw_scores[model] = {
                    'scores': scores_array.tolist(),
                    'statistics': stats
                }
            else:
                raw_scores[model] = {
                    'scores': scores_array.tolist(),
                    'statistics': {'error': 'No valid scores'}
                }
        else:
            raw_scores[model] = {
                'scores': [],
                'statistics': {'error': 'Empty array'}
            }
    
    return jsonify(raw_scores)

@app.route('/api/tuning_results')
def api_tuning_results():
    """API endpoint for hyperparameter tuning results for all models."""
    # Try to load results from file if not in memory
    if not global_results.get('tuning_results'):
        load_results_from_file()
    if not global_results.get('tuning_results'):
        return jsonify({'error': 'No tuning results available', 'message': 'Please run hyperparameter tuning first'})
    
    # Convert to serializable format
    serializable_results = {}
    for model_name, eval_results in global_results['tuning_results'].items():
        serializable_results[model_name] = {}
        for key, value in eval_results.items():
            if hasattr(value, 'tolist'):
                serializable_results[model_name][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[model_name][key] = float(value)
            else:
                serializable_results[model_name][key] = value
    
    return jsonify(serializable_results)

@app.route('/api/comparison_data')
def api_comparison_data():
    """Return comparison and recommendations data."""
    load_results_from_file()
    
    if not global_results.get('comparison') or not global_results.get('recommendations'):
        return jsonify({'error': 'No comparison data available'})
    
    return jsonify({
        'comparison': global_results['comparison'],
        'recommendations': global_results['recommendations']
    })

@app.route('/api/best_model')
def api_best_model():
    """Return the best model information."""
    load_results_from_file()
    
    # First try to get best model from comparison results
    if global_results.get('comparison') and global_results['comparison'].get('recommendations'):
        comparison_recs = global_results['comparison']['recommendations']
        best_model = comparison_recs.get('best_overall', 'No model selected')
        return jsonify({
            'best_model': best_model,
            'best_precision': comparison_recs.get('best_precision', 'N/A'),
            'best_recall': comparison_recs.get('best_recall', 'N/A'),
            'best_f1': comparison_recs.get('best_f1', 'N/A'),
            'best_accuracy': comparison_recs.get('best_accuracy', 'N/A'),
            'best_roc_auc': comparison_recs.get('best_roc_auc', 'N/A')
        })
    
    # Fallback to regular recommendations if comparison not available
    elif global_results.get('recommendations'):
        recommendations = global_results['recommendations']
        # Try to find best model from evaluations
        if global_results.get('evaluations'):
            evaluations = global_results['evaluations']
            # Find best model based on F1 score
            best_model = max(evaluations.keys(), 
                           key=lambda x: evaluations[x].get('f1', 0)) if evaluations else 'No model selected'
        else:
            best_model = 'No model selected'
        
        return jsonify({
            'best_model': best_model,
            'best_precision': 'N/A',
            'best_recall': 'N/A',
            'best_f1': 'N/A',
            'best_accuracy': 'N/A',
            'best_roc_auc': 'N/A'
        })
    
    return jsonify({'error': 'No recommendations available'})

@app.route('/api/debug_state')
def api_debug_state():
    """Debug endpoint to check current state."""
    load_results_from_file()
    
    return jsonify({
        'has_evaluations': bool(global_results.get('evaluations')),
        'evaluations_count': len(global_results.get('evaluations', {})),
        'has_recommendations': bool(global_results.get('recommendations')),
        'has_comparison': bool(global_results.get('comparison')),
        'has_models': bool(global_results.get('models')),
        'models_count': len(global_results.get('models', {})),
        'global_results_keys': list(global_results.keys())
    })



@app.route('/clear_results', methods=['GET', 'POST'])
def clear_results():
    """Clear stored results."""
    import tempfile
    import os
    
    # Clear global results
    global_results.clear()
    global_results.update({
        'data_info': None,
        'models': {},
        'evaluations': {},
        'anomalies': {},
        'tuning_results': {},
        'literature': None,
        'recommendations': None,
        'comparison': None
    })
    
    # Remove stored file
    temp_file = os.path.join(tempfile.gettempdir(), 'anomaly_detection_results.pkl')
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return jsonify({'status': 'success', 'message': 'Results cleared'})

@app.route('/recommendations')
def recommendations():
    """Recommendations page."""
    # Always try to load results from file to ensure data persistence
    load_results_from_file()
    
    if global_results['recommendations'] is None:
        # Check if models have been run
        if not global_results.get('evaluations'):
            # Redirect to index if no models have been run
            return redirect(url_for('index'))
        
        recommendations = Recommendations()
        global_results['recommendations'] = recommendations.generate_recommendations(
            global_results.get('evaluations', {}),
            global_results.get('data_info', {})
        )
    
    return render_template('recommendations.html', 
                         recommendations=global_results['recommendations'])



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict anomalies for new transaction data (CSV upload)."""
    # Load results from file to ensure models are available
    load_results_from_file()
    
    # Debug: Print current state
    print(f"🔍 Predict route - Models available: {bool(global_results.get('models'))}")
    print(f"🔍 Predict route - Models count: {len(global_results.get('models', {}))}")
    print(f"🔍 Predict route - Evaluations available: {bool(global_results.get('evaluations'))}")
    print(f"🔍 Predict route - Evaluations count: {len(global_results.get('evaluations', {}))}")
    
    # Check if models were loaded successfully
    models_loaded = bool(global_results.get('models') and any(global_results.get('models', {}).values()))
    if models_loaded:
        print("✅ Models loaded successfully from saved file")
    else:
        print("⚠️ No models found in memory - will need to retrain if predictions are requested")
    
    if request.method == 'POST':
        # Check if models exist first
        models_available = bool(global_results.get('models') and any(global_results.get('models', {}).values()))
        print(f"🔍 POST request - Models available: {models_available}")
        
        if not models_available:
            return render_template('predict.html', 
                error='No trained models available. Please run models first.')
            
        if 'file' not in request.files:
            return render_template('predict.html', error='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error='No file selected')
        if not file.filename.endswith('.csv'):
            return render_template('predict.html', error='Only CSV files are supported')
        
        # Read uploaded CSV
        df_new = pd.read_csv(file)
        
        # Validate uploaded data
        print(f"🔍 Uploaded data shape: {df_new.shape}")
        print(f"🔍 Uploaded data columns: {list(df_new.columns)}")
        print(f"🔍 Missing values in uploaded data: {df_new.isnull().sum().sum()}")
        
        if df_new.empty:
            return render_template('predict.html', error='Uploaded file is empty')
        
        # Get training data info for feature alignment
        try:
            loader = DataLoader()
            df_train = loader.load_data()
            X_train, _, _ = preprocess_data(df_train, target_column=loader.target_column)
            expected_features = X_train.shape[1]
            print(f"🔍 Expected features from training data: {expected_features}")
        except Exception as e:
            print(f"Error getting training data info: {e}")
            return render_template('predict.html', error='Error accessing training data information')
        
        # Try to get target column from global_results, else infer from uploaded data
        target_col = None
        data_info = global_results.get('data_info') if isinstance(global_results, dict) else None
        if data_info and isinstance(data_info, dict) and 'target_column' in data_info:
            target_col = data_info['target_column']
        else:
            # Try to infer target column from uploaded data
            potential_target_cols = ['class', 'Class', 'target', 'Target', 'label', 'Label', 'fraud', 'Fraud']
            for col in potential_target_cols:
                if col in df_new.columns:
                    target_col = col
                    break
        
        # Preprocess prediction data
        X_new, _, _ = preprocess_data(df_new, target_column=target_col)
        actual_features = X_new.shape[1]
        print(f"🔍 Actual features in prediction data: {actual_features}")
        
        # Feature alignment strategy
        if actual_features != expected_features:
            print(f"⚠️ Feature mismatch detected: Expected {expected_features}, got {actual_features}")
            
            if actual_features < expected_features:
                # Add missing features with zeros
                missing_features = expected_features - actual_features
                print(f"🔧 Adding {missing_features} missing features with zeros")
                X_new = np.column_stack([X_new, np.zeros((X_new.shape[0], missing_features))])
            else:
                # Remove extra features (take first expected_features)
                extra_features = actual_features - expected_features
                print(f"🔧 Removing {extra_features} extra features (keeping first {expected_features})")
                X_new = X_new[:, :expected_features]
            
            print(f"🔧 Feature alignment complete: {X_new.shape}")
        
        # Additional preprocessing for prediction data
        print(f"🔍 Preprocessing prediction data - Final Shape: {X_new.shape}")
        print(f"🔍 NaN values in data: {np.isnan(X_new).sum()}")
        
        # Handle NaN values for models that don't support them
        if np.isnan(X_new).any():
            print("🔍 Handling NaN values in prediction data...")
            # Replace NaN with 0 for models that don't support them
            X_new_clean = np.nan_to_num(X_new, nan=0.0)
            print(f"🔍 NaN values after cleaning: {np.isnan(X_new_clean).sum()}")
        else:
            X_new_clean = X_new
        
        # Run all trained models
        results = {}
        models = global_results.get('models', {}) if isinstance(global_results, dict) else {}
        print(f"Available models for prediction: {list(models.keys())}")
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    # Use cleaned data for models that don't support NaN
                    if name in ['one_class_svm', 'isolation_forest']:
                        preds = model.predict(X_new_clean)
                    else:
                        preds = model.predict(X_new)
                    
                    # Handle different model return types
                    if name in ['autoencoder', 'deep_autoencoder']:
                        # Autoencoders return (preds, errors)
                        if isinstance(preds, tuple) and len(preds) == 2:
                            preds, scores = preds
                        else:
                            scores = preds
                    elif hasattr(model, 'anomaly_score'):
                        # Use cleaned data for anomaly scores too
                        if name in ['one_class_svm', 'isolation_forest']:
                            scores = model.anomaly_score(X_new_clean)
                        else:
                            scores = model.anomaly_score(X_new)
                    else:
                        scores = preds
                    
                    # Format predictions and scores properly
                    # Both preds and scores should be numpy arrays at this point
                    formatted_preds = []
                    formatted_scores = []
                    
                    for i in range(len(preds)):
                        # Convert predictions to integers (0 or 1)
                        pred_val = int(preds[i]) if not np.isnan(preds[i]) else 0
                        
                        # Convert scores to floats
                        score_val = float(scores[i]) if not np.isnan(scores[i]) else 0.0
                        
                        # Check for infinite values
                        if np.isinf(score_val):
                            score_val = 0.0
                        
                        formatted_preds.append(pred_val)
                        formatted_scores.append(score_val)
                    
                    results[name] = {
                        'prediction': formatted_preds,
                        'score': formatted_scores
                    }
                    print(f"✅ {name} predictions completed: {len(formatted_preds)} samples")
            except Exception as e:
                print(f"❌ Error with model {name}: {e}")
                # Provide more specific error messages
                if "NaN" in str(e):
                    results[name] = {'error': f'Data contains missing values that {name} cannot handle'}
                elif "shape" in str(e).lower():
                    results[name] = {'error': f'Data shape mismatch for {name} - expected {expected_features} features, got {actual_features}'}
                else:
                    results[name] = {'error': f'{name} prediction failed: {str(e)}'}
        
        # Summary of prediction results
        successful_models = [name for name, result in results.items() if 'error' not in result]
        failed_models = [name for name, result in results.items() if 'error' in result]
        
        print(f"📊 Prediction Summary:")
        print(f"  ✅ Successful models: {len(successful_models)} - {successful_models}")
        print(f"  ❌ Failed models: {len(failed_models)} - {failed_models}")
        print(f"  📈 Total samples processed: {len(df_new)}")
        
        # Prepare summary for template
        prediction_summary = {
            'successful_count': len(successful_models),
            'failed_count': len(failed_models),
            'failed_models': failed_models,
            'total_samples': len(df_new),
            'feature_alignment': {
                'expected_features': expected_features,
                'actual_features': actual_features,
                'aligned': actual_features == expected_features
            }
        }
        
        return render_template('predict.html', 
                             results=results, 
                             columns=list(df_new.columns), 
                             data=df_new.values.tolist(),
                             prediction_summary=prediction_summary)
    
    # For GET request, check if models are available
    models_available = bool(global_results.get('models') and any(global_results.get('models', {}).values()))
    print(f"🔍 GET request - Models available: {models_available}")
    
    # If no models but evaluations exist, try to re-train
    if not models_available and global_results.get('evaluations'):
        print("🔍 GET request - No models but evaluations exist, attempting re-train...")
        try:
            retrain_models_for_prediction()
            models_available = bool(global_results.get('models') and any(global_results.get('models', {}).values()))
            print(f"🔍 GET request - After re-train, models available: {models_available}")
        except Exception as e:
            print(f"🔍 GET request - Error during re-train: {e}")
    
    return render_template('predict.html', models_available=models_available)

def predict(self, X, threshold=None):
    if hasattr(X, 'vīalues'):
        X = X.values
    errors = self.get_reconstruction_errors(X)
    print("Reconstruction Errors:", errors)  # Debugging
    
    if threshold is None:
        threshold = np.percentile(errors, 95)
    print("Threshold:", threshold)  # Debugging
    
    preds = (errors > threshold).astype(int)
    print("Predictions:", preds)  # Debugging
    
    return preds, errors



if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)