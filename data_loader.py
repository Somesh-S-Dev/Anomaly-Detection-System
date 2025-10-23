import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from visualization import (
    plot_missing_value_heatmap,
    plot_feature_histograms,
    plot_feature_boxplots,
    plot_correlation_matrix
)

class DataLoader:
    def __init__(self, file_path=None):
        self.file_path = file_path or 'SSBCI-Transactions-Dataset.csv'
        self.data = None
        self.target_column = None

    def load_data(self):
        """Load the dataset and identify potential target columns. Also perform feature engineering."""
        self.data = pd.read_csv(self.file_path)
        
        # Check for potential target columns
        potential_target_cols = ['class', 'Class', 'target', 'Target', 'label', 'Label', 'fraud', 'Fraud']
        for col in potential_target_cols:
            if col in self.data.columns:
                self.target_column = col
                break
        
        # --- Feature Engineering ---
        # Add time-based features if 'timestamp' exists
        if 'timestamp' in self.data.columns:
            self.data['transaction_hour'] = pd.to_datetime(self.data['timestamp'], errors='coerce').dt.hour
            self.data['transaction_day'] = pd.to_datetime(self.data['timestamp'], errors='coerce').dt.day
            self.data['transaction_weekday'] = pd.to_datetime(self.data['timestamp'], errors='coerce').dt.weekday
        # Add user-based features if 'user_id' and 'amount' exist
        if 'user_id' in self.data.columns and 'amount' in self.data.columns:
            self.data['user_txn_count'] = self.data.groupby('user_id')['user_id'].transform('count')
            self.data['user_mean_amount'] = self.data.groupby('user_id')['amount'].transform('mean')
            self.data['amt_to_avg'] = self.data['amount'] / (self.data['user_mean_amount'] + 1e-6)
        return self.data

    def show_class_imbalance(self):
        """Show class distribution if target column exists, otherwise show data overview."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        
        if self.target_column:
            class_counts = self.data[self.target_column].value_counts()
            print('Class distribution:')
            print(class_counts)
            plt.figure(figsize=(6,4))
            sns.barplot(x=class_counts.index, y=class_counts.values)
            plt.title('Class Imbalance')
            plt.xlabel(f'{self.target_column} (0=Normal, 1=Anomaly)')
            plt.ylabel('Count')
            plt.show()
        else:
            print('No target column found. This is an unsupervised learning dataset.')
            print(f'Dataset shape: {self.data.shape}')
            print(f'Number of columns: {len(self.data.columns)}')

    def show_basic_stats(self):
        """Show basic statistics of numerical columns."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        
        print('Basic statistics (numerical columns):')
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numerical_cols].describe())

        print(f'\nCategorical columns:')
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f'{col}: {self.data[col].nunique()} unique values')

    def plot_distributions(self, columns=None, max_cols=5):
        """Plot distributions of numerical columns."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if columns is None:
            # Select top numerical columns by variance
            cols = []
            for col in numerical_cols:
                variance = self.data[col].var()
                cols.append((col, variance))
            cols.sort(key=lambda x: x[1], reverse=True)
            cols = [col[0] for col in cols[:max_cols]]
        else:
            cols = [col for col in columns if col in self.data.columns]
        
        for col in cols:
            if col in self.data.columns:
                plt.figure(figsize=(8, 4))
                # Handle infinite values
                data = self.data[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 0:
                    sns.histplot(data.to_numpy(), bins=50, kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(str(col))
                    plt.ylabel('Frequency')
                plt.show()

    def get_numerical_features(self):
        """Get numerical features for anomaly detection."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        return self.data[numerical_cols]

    def get_categorical_features(self):
        """Get categorical features for encoding."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        return self.data[categorical_cols] 

    def show_eda_plots(self):
        """Show EDA plots for the loaded data."""
        if self.data is None:
            raise ValueError('Data not loaded.')
        print("\n--- EDA: Missing Value Heatmap ---")
        plot_missing_value_heatmap(self.data)
        print("\n--- EDA: Feature Histograms ---")
        plot_feature_histograms(self.data)
        print("\n--- EDA: Feature Boxplots ---")
        plot_feature_boxplots(self.data)
        print("\n--- EDA: Correlation Matrix ---")
        plot_correlation_matrix(self.data) 