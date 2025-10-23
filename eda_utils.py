import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda_pipeline(csv_path):
    df = pd.read_csv(csv_path)

    # Drop columns with >50% missing values
    missing_percent = df.isnull().mean()
    dropped_columns = missing_percent[missing_percent > 0.5].index.tolist()
    df_cleaned = df.drop(columns=dropped_columns)

    # Fill missing values in specified categorical columns
    cat_cols = [
        'lender_name', 'lender_type', 'lender_type_category',
        'optional_woman_owned', 'optional_minority_owned', 'optional_business_city'
    ]
    for col in cat_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')

    # Parse 'disbursement_date' to datetime and drop invalid rows
    if 'disbursement_date' in df_cleaned.columns:
        df_cleaned['disbursement_date'] = pd.to_datetime(df_cleaned['disbursement_date'], errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['disbursement_date'])

    # Create directory for EDA images
    img_dir = os.path.join('static', 'eda')
    os.makedirs(img_dir, exist_ok=True)
    plot_paths = {}

    # Distribution of Loan/Investment Amount
    plt.figure(figsize=(12, 6))
    sns.histplot(df_cleaned['loan_investment_amount'], bins=50, kde=True)
    plt.title('Distribution of Loan/Investment Amount', fontsize=16)
    plt.xlabel('Amount', fontsize=12)
    plt.ylabel('Frequency (Log Scale)', fontsize=12)
    plt.yscale('log')
    loan_dist_path = os.path.join(img_dir, 'loan_investment_amount.png')
    plt.tight_layout()
    plt.savefig(loan_dist_path)
    plt.close()
    plot_paths['loan_investment_amount'] = loan_dist_path

    # Categorical Counts
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    sns.countplot(y=df_cleaned['program_type'], ax=axes[0], order=df_cleaned['program_type'].value_counts().index)
    axes[0].set_title('Count by Program Type', fontsize=14)
    sns.countplot(y=df_cleaned['lender_type'], ax=axes[1], order=df_cleaned['lender_type'].value_counts().index)
    axes[1].set_title('Count by Lender Type', fontsize=14)
    sns.countplot(x=df_cleaned['metro_type'], ax=axes[2], order=df_cleaned['metro_type'].value_counts().index)
    axes[2].set_title('Count by Metro Type', fontsize=14)
    plt.tight_layout()
    cat_counts_path = os.path.join(img_dir, 'categorical_counts.png')
    plt.savefig(cat_counts_path)
    plt.close()
    plot_paths['categorical_counts'] = cat_counts_path

    # Correlation Matrix
    plt.figure(figsize=(18, 15))
    numeric_cols = df_cleaned.select_dtypes(include='number')
    correlation_matrix = numeric_cols.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    corr_matrix_path = os.path.join(img_dir, 'correlation_matrix.png')
    plt.tight_layout()
    plt.savefig(corr_matrix_path)
    plt.close()
    plot_paths['correlation_matrix'] = corr_matrix_path

    # Missing Value Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_cleaned.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Value Heatmap')
    missing_heatmap_path = os.path.join(img_dir, 'missing_heatmap.png')
    plt.tight_layout()
    plt.savefig(missing_heatmap_path)
    plt.close()
    plot_paths['missing_heatmap'] = missing_heatmap_path

    # Prepare summary dict with all expected keys
    summary = {
        'shape': df_cleaned.shape,
        'columns': list(df_cleaned.columns),
        'dtypes': {col: str(dtype) for col, dtype in df_cleaned.dtypes.items()},
        'missing_percent': [
            f"{col}: {100 * pct:.2f}%" for col, pct in df_cleaned.isnull().mean().items()
        ],
        'describe': df_cleaned.describe(include='all').to_html(classes="table table-bordered table-sm"),
        'eda_images': plot_paths,
        'sample_data': df_cleaned.head(10).to_dict(orient='records')  # <-- Add this line
    }
    return df_cleaned, summary