# Financial Anomaly Detection System

A comprehensive anomaly detection system for financial fraud detection with both Flask web interface and enhanced terminal interface, implementing multiple machine learning models and providing detailed analysis and recommendations.

## 🎯 Objectives Achieved

### O1: Literature Research on Fraud Detection Techniques
- ✅ Comprehensive literature review of statistical, ML, deep learning, and ensemble methods
- ✅ Analysis of different fraud types (credit card, insurance, banking, investment)
- ✅ Evaluation metrics comparison and best practices
- ✅ Recent research findings and trends

### O2: Machine Learning Models Implementation
- ✅ **Isolation Forest**: Ensemble-based anomaly detection
- ✅ **One-Class SVM**: Support vector machine for outlier detection
- ✅ **Autoencoder**: Deep learning neural network for pattern recognition
- ✅ **Deep Autoencoder**: Enhanced neural network with multiple layers
- ✅ **Random Forest**: Supervised ensemble method for classification
- ✅ **XGBoost**: Gradient boosting for high-performance classification
- ✅ PyTorch implementation with GPU support

### O3: Dataset Preprocessing and Training
- ✅ SSBCI-Transactions-Dataset.csv integration
- ✅ Robust data preprocessing pipeline with advanced techniques
- ✅ Missing value handling and feature engineering
- ✅ Categorical encoding and numerical scaling
- ✅ Class imbalance handling with SMOTE, ADASYN, and other techniques
- ✅ Automatic resampling strategy selection

### O4: Model Evaluation and Testing
- ✅ Comprehensive evaluation metrics (Precision, Recall, F1, ROC AUC, PR AUC)
- ✅ Real-time model comparison dashboard
- ✅ Enhanced terminal interface with detailed results display
- ✅ Anomaly score analysis and visualization
- ✅ Performance benchmarking with train/test splits
- ✅ Hyperparameter tuning with GridSearch and RandomizedSearch

### O5: Analysis and Recommendations
- ✅ Policy recommendations for legislators
- ✅ Technical recommendations for financial institutions
- ✅ Regulatory framework suggestions
- ✅ Cost-benefit analysis and risk assessment

## 🚀 Features

### Web Dashboard
- **Interactive Flask UI** with real-time model comparison
- **Literature Review** section with comprehensive research
- **Data Analysis** with statistical summaries and visualizations
- **Model Comparison** with detailed performance metrics
- **Recommendations** for policymakers and institutions
- **File Upload** capability for custom datasets
- **Hyperparameter Tuning** interface for model optimization

### Enhanced Terminal Interface
- **Detailed Model Results**: Individual performance metrics for each model
- **Formatted Output**: Clean, organized display with emojis and separators
- **Confusion Matrices**: ASCII art representation of predictions
- **Training Time Tracking**: Performance monitoring for each model
- **Comparison Tables**: Side-by-side model performance comparison
- **Best Model Identification**: Automatic highlighting of top performers

### Machine Learning Models
- **Isolation Forest**: Fast, scalable ensemble method
- **One-Class SVM**: Robust kernel-based approach
- **Autoencoder**: Deep learning with PyTorch
- **Deep Autoencoder**: Enhanced neural network architecture
- **Random Forest**: Supervised ensemble classification
- **XGBoost**: High-performance gradient boosting
- **Ensemble Methods**: Combined model predictions

### Analysis Tools
- **Real-time Visualization**: Charts and graphs
- **Performance Metrics**: Comprehensive evaluation
- **Risk Assessment**: Automated risk analysis
- **Cost-Benefit Analysis**: ROI calculations
- **SHAP Analysis**: Model interpretability
- **Advanced Preprocessing**: Multiple resampling strategies

## 📊 Dataset

The system uses the **SSBCI-Transactions-Dataset.csv** containing:
- 21,962 transactions
- 49 features including financial and demographic data
- Government financial transaction records
- Suitable for both supervised and unsupervised anomaly detection

## 🧹 Preprocessing & EDA Plots

This project provides comprehensive Exploratory Data Analysis (EDA) visualizations to guide preprocessing and feature engineering decisions. All plots are generated in the backend and displayed in the dashboard.

### 1. Missing Value Heatmap
**Purpose:** Visualizes missing values in the dataset to identify columns/rows with many missing entries.

```python
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.tight_layout()
plt.savefig('static/eda/missing_heatmap.png')
plt.close()
```
*Bright colors indicate missing values. Helps decide on imputation or removal strategies.*

### 2. Feature Histograms
**Purpose:** Shows the distribution of each numerical feature, revealing skewness, outliers, and the general shape of the data.

```python
def plot_feature_histogram(df, col, save_path):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f'Histogram: {col}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```
*One image per feature. Useful for detecting outliers and understanding scaling needs.*

### 3. Feature Boxplots
**Purpose:** Visualizes the spread, median, and outliers for each numerical feature.

```python
def plot_feature_boxplot(df, col, save_path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col].dropna())
    plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```
*One image per feature. Quickly spot features with extreme values or skewed distributions.*

### 4. Advanced Preprocessing Features
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **BorderlineSMOTE**: Borderline-aware oversampling
- **RandomUnderSampler**: Random undersampling
- **TomekLinks**: Tomek links cleaning
- **SMOTETomek/SMOTEENN**: Combined resampling techniques

All these plots are available in the dashboard and help guide preprocessing, feature engineering, and model selection decisions.

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Installation
```bash
# Clone or download the project
cd Anamoly Detection

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py

# Or run the terminal interface
python main.py
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Install other dependencies
pip install flask pandas numpy scikit-learn matplotlib seaborn plotly werkzeug shap imbalanced-learn joblib xgboost

# Run the application
python app.py
```

## 🎮 Usage

### Starting the Application
```bash
# Web Interface
python app.py

# Terminal Interface
python main.py
```

The web application will be available at `http://localhost:5000`

### Web Interface Navigation
1. **Dashboard**: Overview and model execution
2. **Literature Review**: Research findings and techniques
3. **Data Analysis**: Dataset exploration and statistics
4. **Model Comparison**: Performance analysis and charts
5. **Recommendations**: Policy and technical guidance
6. **Upload Data**: Custom dataset integration
7. **Hyperparameter Tuning**: Model optimization interface

### Terminal Interface Features
The enhanced terminal interface provides:
- **Individual Model Results**: Detailed metrics for each model
- **Formatted Display**: Clean, organized output with visual separators
- **Performance Comparison**: Side-by-side model comparison tables
- **Training Time Tracking**: Performance monitoring
- **Confusion Matrices**: Visual prediction analysis
- **Best Model Identification**: Automatic highlighting of top performers

### Running Models
1. **Web Interface**: Navigate to Dashboard and click "Run Models"
2. **Terminal Interface**: Run `python main.py` for comprehensive analysis
3. **Hyperparameter Tuning**: Use the web interface for model optimization

## 📈 Model Performance

### Evaluation Metrics
- **Precision**: Accuracy of anomaly detection
- **Recall**: Coverage of actual anomalies
- **F1-Score**: Balanced performance measure
- **ROC AUC**: Overall model discrimination
- **PR AUC**: Performance on imbalanced data
- **Accuracy**: Overall classification accuracy

### Model Characteristics
| Model | Type | Pros | Cons | Best For |
|-------|------|------|------|----------|
| Isolation Forest | Ensemble | Fast, scalable, no tuning | Random nature, limited interpretability | Large datasets |
| One-Class SVM | Kernel-based | Good generalization, flexible | Parameter sensitive, computational cost | Medium datasets |
| Autoencoder | Neural Network | Excellent feature learning, state-of-the-art | High computational cost, black box | Complex patterns |
| Deep Autoencoder | Neural Network | Enhanced feature learning, multiple layers | Very high computational cost | Very complex patterns |
| Random Forest | Supervised | Good interpretability, robust | Requires labeled data | Balanced datasets |
| XGBoost | Supervised | High performance, feature importance | Requires labeled data, parameter tuning | High-performance needs |

### Performance Comparison
The system now includes comprehensive performance comparison with:
- **Default vs Tuned Models**: Side-by-side comparison
- **Training Time Analysis**: Performance monitoring
- **Best Model Identification**: Automatic selection
- **Detailed Metrics**: All evaluation criteria

## 🔬 Research Analysis

### Literature Review Findings
- **Statistical Methods**: Simple but limited to linear relationships
- **Machine Learning**: Good balance of performance and interpretability
- **Deep Learning**: Best performance but requires large datasets
- **Ensemble Methods**: Robust performance with reduced overfitting
- **Recent Advances**: GANs, transformers, and federated learning

### Fraud Types Analysis
- **Credit Card Fraud**: Real-time monitoring and behavioral analysis
- **Insurance Fraud**: Claim analysis and social network detection
- **Banking Fraud**: KYC procedures and transaction monitoring
- **Investment Fraud**: Market surveillance and pattern recognition

## 💡 Recommendations

### For Policymakers
- Establish secure data sharing protocols
- Develop AI/ML standards for fraud detection
- Provide funding for advanced research
- Implement regulatory frameworks for AI systems

### For Financial Institutions
- Implement real-time transaction monitoring
- Use ensemble methods for robust detection
- Educate customers about fraud prevention
- Invest in advanced ML infrastructure

### For Regulators
- Establish minimum fraud detection requirements
- Require regular system audits
- Coordinate international cooperation
- Develop AI governance frameworks

## 🛡️ Security & Privacy

- **Local Processing**: All data processed locally
- **No Permanent Storage**: Files deleted after processing
- **Secure Validation**: File type and content validation
- **Session-based Results**: Temporary result storage
- **Input Sanitization**: Protection against malicious inputs

## 📁 Project Structure

```
version7/
├── app.py                          # Flask application
├── main.py                         # Enhanced terminal interface
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── research_analysis.py            # Literature review and recommendations
├── data_loader.py                  # Data loading and exploration
├── preprocessing.py                # Advanced data preprocessing pipeline
├── evaluation.py                   # Model evaluation metrics
├── visualization.py                # Plotting and charts
├── hyperparameter_tuning.py        # Hyperparameter tuning logic
├── models/                         # ML model implementations
│   ├── autoencoder.py              # PyTorch autoencoder
│   ├── isolation_forest.py         # Scikit-learn isolation forest
│   ├── one_class_svm.py            # Scikit-learn one-class SVM
│   ├── random_forest_model.py      # Scikit-learn random forest
│   └── xgboost_model.py            # XGBoost model
├── templates/                      # Flask HTML templates
│   ├── base.html                   # Base template
│   ├── index.html                  # Dashboard
│   ├── literature.html             # Literature review
│   ├── data_analysis.html          # Data analysis
│   ├── model_comparison.html       # Model comparison
│   ├── recommendations.html        # Recommendations
│   ├── upload.html                 # File upload
│   └── eda_plots.html              # EDA visualizations
├── static/                         # Static files and generated plots
│   └── eda/                        # EDA visualization images
├── uploads/                        # File upload directory
└── SSBCI-Transactions-Dataset.csv  # Financial dataset
```

## 🔧 Technical Details

### Dependencies
- **Flask**: Web framework
- **PyTorch**: Deep learning framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Plotly**: Interactive charts
- **SHAP**: Model interpretability
- **XGBoost**: Gradient boosting
- **Imbalanced-learn**: Class imbalance handling
- **Joblib**: Model persistence

### Model Architecture
- **Autoencoder**: Encoder-Decoder with ReLU activation
- **Deep Autoencoder**: Multi-layer encoder-decoder
- **Isolation Forest**: Random forest ensemble
- **One-Class SVM**: RBF kernel with optimized parameters
- **Random Forest**: Supervised ensemble with feature importance
- **XGBoost**: Gradient boosting with regularization

### Performance Optimization
- **GPU Support**: Automatic CUDA detection
- **Batch Processing**: Efficient data handling
- **Memory Management**: Optimized for large datasets
- **Real-time Updates**: Live dashboard updates
- **Parallel Processing**: Multi-core utilization
- **Caching**: Result caching for faster access

## 📊 Results and Analysis

### Model Performance Comparison
- **Best Overall**: Deep Autoencoder (enhanced neural network)
- **Best Precision**: One-Class SVM (kernel-based approach)
- **Best Recall**: Random Forest (supervised ensemble)
- **Most Robust**: Ensemble combination
- **Fastest**: Isolation Forest (unsupervised)

### Risk Assessment
- **Low Risk**: F1-score ≥ 0.9
- **Medium Risk**: F1-score 0.7-0.9
- **High Risk**: F1-score < 0.7

### Cost-Benefit Analysis
- **ROI**: 150-300% depending on fraud prevention rate
- **Implementation Cost**: $500,000 estimated
- **Annual Savings**: $1-5 million depending on scale
- **Maintenance Cost**: $50,000 annually

## 🚀 Future Enhancements

### Planned Features
- **Real-time Streaming**: Live transaction monitoring
- **Advanced Models**: GANs and transformer-based approaches
- **API Integration**: RESTful API for external systems
- **Mobile App**: iOS/Android companion app
- **Cloud Deployment**: AWS/Azure integration
- **Federated Learning**: Privacy-preserving distributed training

### Research Directions
- **Federated Learning**: Privacy-preserving distributed training
- **Explainable AI**: Model interpretability improvements
- **Adversarial Training**: Robustness against evasion attacks
- **Multi-modal Detection**: Text, image, and transaction analysis
- **Quantum ML**: Quantum computing for fraud detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `/docs` folder

---

**Note**: This system is designed for research and educational purposes. For production use in financial institutions, additional security measures and regulatory compliance should be implemented.
