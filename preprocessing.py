import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import warnings
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df):
    # Add time-based features if 'timestamp' exists
    if 'timestamp' in df.columns:
        df['transaction_hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
        df['transaction_day'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.day
        df['transaction_weekday'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.weekday
    # Add user-based features if 'user_id' and 'amount' exist
    if 'user_id' in df.columns and 'amount' in df.columns:
        df['user_txn_count'] = df.groupby('user_id')['user_id'].transform('count')
        df['user_mean_amount'] = df.groupby('user_id')['amount'].transform('mean')
        df['amt_to_avg'] = df['amount'] / (df['user_mean_amount'] + 1e-6)
    return df

def analyze_class_imbalance(y):
    """
    Analyze class distribution and provide imbalance statistics.
    
    Args:
        y: Target labels
        
    Returns:
        dict: Imbalance analysis results
    """
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    
    # Calculate imbalance ratios
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]
    majority_count = np.max(counts)
    minority_count = np.min(counts)
    
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
    
    analysis = {
        'total_samples': total_samples,
        'majority_class': majority_class,
        'minority_class': minority_class,
        'majority_count': majority_count,
        'minority_count': minority_count,
        'imbalance_ratio': imbalance_ratio,
        'minority_percentage': (minority_count / total_samples) * 100,
        'severity': 'balanced' if imbalance_ratio < 2 else 
                   'slight' if imbalance_ratio < 10 else
                   'moderate' if imbalance_ratio < 100 else 'severe'
    }
    
    print(f"🔍 Class Imbalance Analysis:")
    print(f"   Total samples: {total_samples}")
    print(f"   Majority class ({majority_class}): {majority_count} samples")
    print(f"   Minority class ({minority_class}): {minority_count} samples")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"   Minority percentage: {analysis['minority_percentage']:.2f}%")
    print(f"   Severity: {analysis['severity']}")
    
    return analysis

def apply_resampling_strategy(X, y, strategy='auto', random_state=42):
    """
    Apply appropriate resampling strategy based on class imbalance analysis.
    
    Args:
        X: Feature matrix
        y: Target labels
        strategy: Resampling strategy ('auto', 'smote', 'adasyn', 'borderline_smote', 
                 'random_under', 'tomek', 'enn', 'smote_tomek', 'smote_enn', 'none')
        random_state: Random seed
        
    Returns:
        tuple: (X_resampled, y_resampled, resampling_info)
    """
    if strategy == 'none':
        return X, y, {'method': 'none', 'original_shape': X.shape}
    
    # Analyze imbalance first
    imbalance_analysis = analyze_class_imbalance(y)
    
    # Auto-strategy selection based on imbalance severity
    if strategy == 'auto':
        if imbalance_analysis['severity'] == 'balanced':
            strategy = 'none'
        elif imbalance_analysis['severity'] == 'slight':
            strategy = 'smote'
        elif imbalance_analysis['severity'] == 'moderate':
            strategy = 'borderline_smote'
        elif imbalance_analysis['severity'] == 'severe':
            strategy = 'smote_enn'
    
    print(f"🔄 Applying resampling strategy: {strategy}")
    
    try:
        if strategy == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=min(5, imbalance_analysis['minority_count']-1))
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'SMOTE'
            
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=random_state, n_neighbors=min(5, imbalance_analysis['minority_count']-1))
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'ADASYN'
            
        elif strategy == 'borderline_smote':
            sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=min(5, imbalance_analysis['minority_count']-1))
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'BorderlineSMOTE'
            
        elif strategy == 'random_under':
            sampler = RandomUnderSampler(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'RandomUnderSampler'
            
        elif strategy == 'tomek':
            sampler = TomekLinks()
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'TomekLinks'
            
        elif strategy == 'enn':
            sampler = EditedNearestNeighbours()
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'EditedNearestNeighbours'
            
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'SMOTE + TomekLinks'
            
        elif strategy == 'smote_enn':
            sampler = SMOTEENN(random_state=random_state)
            X_res, y_res = sampler.fit_resample(X, y)
            method = 'SMOTE + ENN'
            
        else:
            return X, y, {'method': 'none', 'original_shape': X.shape}
        
        # Analyze results
        new_analysis = analyze_class_imbalance(y_res)
        
        resampling_info = {
            'method': method,
            'original_shape': X.shape,
            'resampled_shape': X_res.shape,
            'original_imbalance': imbalance_analysis,
            'resampled_imbalance': new_analysis,
            'samples_added': X_res.shape[0] - X.shape[0],
            'samples_removed': X.shape[0] - X_res.shape[0]
        }
        
        print(f"✅ Resampling completed: {method}")
        print(f"   Original shape: {X.shape}")
        print(f"   Resampled shape: {X_res.shape}")
        print(f"   Samples added: {resampling_info['samples_added']}")
        print(f"   Samples removed: {resampling_info['samples_removed']}")
        
        return X_res, y_res, resampling_info
        
    except Exception as e:
        print(f"❌ Resampling failed: {e}")
        print("   Falling back to original data")
        return X, y, {'method': 'failed', 'error': str(e), 'original_shape': X.shape}

# Update preprocess_data to use advanced_feature_engineering and robust preprocessing

def preprocess_data(df, apply_smote=False, target_column=None, resampling_strategy='auto', random_state=42):
    """
    Preprocess data for anomaly detection with advanced feature engineering and robust preprocessing.
    
    Args:
        df: Input dataframe
        apply_smote: Deprecated - use resampling_strategy instead
        target_column: Name of target column
        resampling_strategy: Resampling strategy for handling class imbalance
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X, y, preprocessing_info)
    """
    df_processed = df.copy()
    df_processed = advanced_feature_engineering(df_processed)
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col].fillna('Unknown', inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encode categoricals
    for col in df_processed.select_dtypes(include='object').columns:
        df_processed[col] = LabelEncoder().fit_transform(df_processed[col])
    
    # Separate features and target
    if target_column and target_column in df_processed.columns:
        y = df_processed[target_column].values
        X = df_processed.drop(columns=[target_column]).values
    else:
        y = None
        X = df_processed.values
    
    preprocessing_info = {
        'original_shape': df.shape,
        'processed_shape': X.shape,
        'target_column': target_column,
        'has_target': y is not None,
        'resampling_applied': False
    }
    
    # Apply resampling if target exists and strategy is specified
    if y is not None and resampling_strategy != 'none':
        X, y, resampling_info = apply_resampling_strategy(X, y, resampling_strategy, random_state)
        preprocessing_info.update(resampling_info)
        preprocessing_info['resampling_applied'] = True
    
    return X, y, preprocessing_info

def create_anomaly_labels(X, contamination=0.01, random_state=42):
    """
    Create synthetic anomaly labels for evaluation purposes.
    This is useful when no target column exists.
    
    Args:
        X: Feature matrix
        contamination: Fraction of anomalies to create
        random_state: Random seed for reproducible results
    
    Returns:
        y: Synthetic anomaly labels
    """
    # Set random seed for reproducible results
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_anomalies = int(n_samples * contamination)
    
    # Create synthetic anomalies by adding noise to random samples
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    y = np.zeros(n_samples)
    y[anomaly_indices] = 1
    
    print(f"Created synthetic anomalies: {n_anomalies} out of {n_samples} samples ({contamination*100:.1f}%)")
    
    # Analyze the created imbalance
    analyze_class_imbalance(y)
    
    return y