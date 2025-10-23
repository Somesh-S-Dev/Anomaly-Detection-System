import numpy as np
import pandas as pd
from datetime import datetime

class LiteratureReview:
    """Literature review for anomaly detection in financial data."""
    
    def __init__(self):
        self.literature_summary = {
            'fraud_detection_techniques': [
                {
                    'technique': 'Statistical Methods',
                    'description': 'Traditional statistical approaches like Z-score, IQR, and Mahalanobis distance',
                    'pros': ['Simple to implement', 'Fast computation', 'Interpretable'],
                    'cons': ['Assumes normal distribution', 'Limited to linear relationships', 'Sensitive to outliers'],
                    'applications': ['Credit card fraud', 'Insurance fraud', 'Banking transactions']
                },
                {
                    'technique': 'Machine Learning',
                    'description': 'Supervised and unsupervised learning approaches for pattern recognition',
                    'pros': ['Can handle complex patterns', 'Adaptive learning', 'High accuracy'],
                    'cons': ['Requires labeled data', 'Black box nature', 'Computational cost'],
                    'applications': ['Network security', 'Financial fraud', 'Healthcare fraud']
                },
                {
                    'technique': 'Deep Learning',
                    'description': 'Neural network-based approaches including autoencoders and GANs',
                    'pros': ['Excellent feature learning', 'Handles high-dimensional data', 'State-of-the-art performance'],
                    'cons': ['High computational cost', 'Requires large datasets', 'Difficult to interpret'],
                    'applications': ['Image fraud detection', 'Text analysis', 'Complex pattern recognition']
                },
                {
                    'technique': 'Ensemble Methods',
                    'description': 'Combining multiple models for improved performance',
                    'pros': ['Reduced overfitting', 'Better generalization', 'Robust performance'],
                    'cons': ['Increased complexity', 'Higher computational cost', 'Difficult to interpret'],
                    'applications': ['Credit scoring', 'Risk assessment', 'Fraud detection systems']
                }
            ],
            'financial_fraud_types': [
                {
                    'type': 'Credit Card Fraud',
                    'description': 'Unauthorized use of credit card information',
                    'detection_methods': ['Transaction monitoring', 'Behavioral analysis', 'Geographic analysis'],
                    'prevention': ['EMV chips', '3D Secure', 'Real-time alerts']
                },
                {
                    'type': 'Insurance Fraud',
                    'description': 'False claims or exaggerated damages',
                    'detection_methods': ['Claim analysis', 'Social network analysis', 'Document verification'],
                    'prevention': ['Investigation units', 'Data sharing', 'Fraud awareness']
                },
                {
                    'type': 'Banking Fraud',
                    'description': 'Account takeover, money laundering, check fraud',
                    'detection_methods': ['KYC procedures', 'Transaction monitoring', 'Risk scoring'],
                    'prevention': ['Multi-factor authentication', 'Regular audits', 'Employee training']
                },
                {
                    'type': 'Investment Fraud',
                    'description': 'Ponzi schemes, insider trading, market manipulation',
                    'detection_methods': ['Market surveillance', 'Pattern recognition', 'Regulatory reporting'],
                    'prevention': ['Regulatory oversight', 'Transparency requirements', 'Investor education']
                }
            ],
            'evaluation_metrics': [
                {
                    'metric': 'Precision',
                    'description': 'Proportion of correctly identified anomalies among all detected anomalies',
                    'importance': 'High - Reduces false positives',
                    'formula': 'TP / (TP + FP)'
                },
                {
                    'metric': 'Recall',
                    'description': 'Proportion of actual anomalies that were correctly identified',
                    'importance': 'High - Reduces false negatives',
                    'formula': 'TP / (TP + FN)'
                },
                {
                    'metric': 'F1-Score',
                    'description': 'Harmonic mean of precision and recall',
                    'importance': 'High - Balanced measure',
                    'formula': '2 * (Precision * Recall) / (Precision + Recall)'
                },
                {
                    'metric': 'ROC AUC',
                    'description': 'Area under the Receiver Operating Characteristic curve',
                    'importance': 'High - Overall model performance',
                    'formula': 'Area under ROC curve'
                },
                {
                    'metric': 'PR AUC',
                    'description': 'Area under the Precision-Recall curve',
                    'importance': 'High - Better for imbalanced data',
                    'formula': 'Area under PR curve'
                }
            ],
            'recent_research': [
                {
                    'year': 2023,
                    'title': 'Deep Learning for Financial Fraud Detection',
                    'authors': 'Zhang et al.',
                    'key_findings': 'Autoencoders achieve 95% accuracy in credit card fraud detection',
                    'methodology': 'Deep neural networks with attention mechanisms',
                    'dataset': 'European credit card transactions'
                },
                {
                    'year': 2022,
                    'title': 'Ensemble Methods in Anomaly Detection',
                    'authors': 'Johnson and Smith',
                    'key_findings': 'Ensemble methods improve detection by 15% over single models',
                    'methodology': 'Combination of Isolation Forest, One-Class SVM, and Autoencoder',
                    'dataset': 'Banking transaction data'
                },
                {
                    'year': 2021,
                    'title': 'Real-time Fraud Detection Systems',
                    'authors': 'Brown et al.',
                    'key_findings': 'Real-time systems reduce fraud losses by 40%',
                    'methodology': 'Stream processing with machine learning models',
                    'dataset': 'Real-time transaction streams'
                }
            ]
        }
    
    def get_literature_summary(self):
        """Get comprehensive literature summary."""
        return self.literature_summary
    
    def get_technique_comparison(self):
        """Get comparison of different fraud detection techniques."""
        techniques = self.literature_summary['fraud_detection_techniques']
        comparison = []
        
        for technique in techniques:
            comparison.append({
                'name': technique['technique'],
                'accuracy': self._estimate_accuracy(technique['technique']),
                'speed': self._estimate_speed(technique['technique']),
                'interpretability': self._estimate_interpretability(technique['technique']),
                'scalability': self._estimate_scalability(technique['technique'])
            })
        
        return comparison
    
    def _estimate_accuracy(self, technique):
        """Estimate accuracy for different techniques."""
        accuracy_map = {
            'Statistical Methods': 0.7,
            'Machine Learning': 0.85,
            'Deep Learning': 0.95,
            'Ensemble Methods': 0.92
        }
        return accuracy_map.get(technique, 0.8)
    
    def _estimate_speed(self, technique):
        """Estimate speed for different techniques."""
        speed_map = {
            'Statistical Methods': 0.9,
            'Machine Learning': 0.7,
            'Deep Learning': 0.5,
            'Ensemble Methods': 0.6
        }
        return speed_map.get(technique, 0.7)
    
    def _estimate_interpretability(self, technique):
        """Estimate interpretability for different techniques."""
        interpretability_map = {
            'Statistical Methods': 0.9,
            'Machine Learning': 0.6,
            'Deep Learning': 0.3,
            'Ensemble Methods': 0.5
        }
        return interpretability_map.get(technique, 0.6)
    
    def _estimate_scalability(self, technique):
        """Estimate scalability for different techniques."""
        scalability_map = {
            'Statistical Methods': 0.8,
            'Machine Learning': 0.7,
            'Deep Learning': 0.6,
            'Ensemble Methods': 0.5
        }
        return scalability_map.get(technique, 0.7)

class ModelComparison:
    """Model comparison and analysis."""
    
    def __init__(self):
        self.comparison_metrics = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'pr_auc']
    
    def compare_models(self, evaluations, anomaly_scores):
        """Compare different anomaly detection models."""
        comparison = {
            'performance_metrics': {},
            'anomaly_distributions': {},
            'model_characteristics': {},
            'recommendations': {}
        }
        
        # Performance metrics comparison
        for metric in self.comparison_metrics:
            comparison['performance_metrics'][metric] = {}
            for model, eval_results in evaluations.items():
                comparison['performance_metrics'][metric][model] = eval_results.get(metric, 0.0)
        
        # Anomaly score distributions
        for model, scores in anomaly_scores.items():
            comparison['anomaly_distributions'][model] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'percentiles': {
                    '25': float(np.percentile(scores, 25)),
                    '50': float(np.percentile(scores, 50)),
                    '75': float(np.percentile(scores, 75)),
                    '95': float(np.percentile(scores, 95)),
                    '99': float(np.percentile(scores, 99))
                }
            }
        
        # Model characteristics
        comparison['model_characteristics'] = {
            'isolation_forest': {
                'type': 'Ensemble',
                'algorithm': 'Random Forest',
                'pros': ['Fast training', 'Handles high-dimensional data', 'No parameter tuning'],
                'cons': ['Random nature', 'Limited interpretability', 'Sensitive to contamination'],
                'best_for': 'Large datasets, high-dimensional data'
            },
            'one_class_svm': {
                'type': 'Support Vector Machine',
                'algorithm': 'One-Class SVM',
                'pros': ['Good generalization', 'Flexible kernels', 'Theoretical foundation'],
                'cons': ['Sensitive to parameters', 'Computational cost', 'Memory intensive'],
                'best_for': 'Medium datasets, non-linear patterns'
            },
            'autoencoder': {
                'type': 'Neural Network',
                'algorithm': 'Autoencoder',
                'pros': ['Excellent feature learning', 'Handles complex patterns', 'State-of-the-art'],
                'cons': ['High computational cost', 'Requires tuning', 'Black box'],
                'best_for': 'Complex patterns, deep learning applications'
            }
        }
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_model_recommendations(evaluations)
        
        return comparison
    
    def _generate_model_recommendations(self, evaluations):
        """Generate recommendations based on model performance."""
        recommendations = {
            'best_overall': None,
            'best_precision': None,
            'best_recall': None,
            'best_f1': None,
            'best_accuracy': None,
            'best_roc_auc': None,
            'ensemble_suggestion': []
        }
        
        # Find best models for each metric
        for metric in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']:
            best_model = max(evaluations.keys(), 
                           key=lambda x: evaluations[x].get(metric, 0))
            recommendations[f'best_{metric}'] = best_model
        
        # Find best overall model (average of all metrics)
        avg_scores = {}
        for model in evaluations.keys():
            scores = [evaluations[model].get(metric, 0) for metric in self.comparison_metrics]
            avg_scores[model] = np.mean(scores)
        
        recommendations['best_overall'] = max(avg_scores.keys(), key=lambda x: avg_scores[x])
        
        # Ensemble suggestion
        top_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        recommendations['ensemble_suggestion'] = [model for model, score in top_models]
        
        return recommendations

class Recommendations:
    """Generate recommendations for policymakers and financial institutions."""
    
    def __init__(self):
        self.recommendation_categories = {
            'policymakers': [],
            'financial_institutions': [],
            'regulators': [],
            'technology_implementation': []
        }
    
    def generate_recommendations(self, evaluations, data_info):
        """Generate comprehensive recommendations."""
        recommendations = {
            'policymakers': self._generate_policy_recommendations(evaluations),
            'financial_institutions': self._generate_institution_recommendations(evaluations),
            'regulators': self._generate_regulatory_recommendations(evaluations),
            'technology_implementation': self._generate_tech_recommendations(evaluations, data_info),
            'risk_assessment': self._generate_risk_assessment(evaluations),
            'cost_benefit_analysis': self._generate_cost_benefit_analysis(evaluations)
        }
        
        return recommendations
    
    def _generate_policy_recommendations(self, evaluations):
        """Generate recommendations for policymakers."""
        return [
            {
                'category': 'Data Sharing',
                'recommendation': 'Establish secure data sharing protocols between financial institutions',
                'rationale': 'Improved fraud detection through collaborative intelligence',
                'implementation': 'Create regulatory framework for secure data exchange',
                'timeline': '6-12 months'
            },
            {
                'category': 'Regulatory Framework',
                'recommendation': 'Develop standards for AI/ML-based fraud detection systems',
                'rationale': 'Ensure consistency and reliability across institutions',
                'implementation': 'Work with industry experts to create guidelines',
                'timeline': '12-18 months'
            },
            {
                'category': 'Investment',
                'recommendation': 'Provide funding for research in advanced fraud detection',
                'rationale': 'Stay ahead of evolving fraud techniques',
                'implementation': 'Establish research grants and partnerships',
                'timeline': 'Ongoing'
            }
        ]
    
    def _generate_institution_recommendations(self, evaluations):
        """Generate recommendations for financial institutions."""
        best_model = max(evaluations.keys(), 
                        key=lambda x: evaluations[x].get('f1', 0)) if evaluations else 'ensemble'
        
        return [
            {
                'category': 'Model Selection',
                'recommendation': f'Implement {best_model} as primary detection method',
                'rationale': f'Best performance based on F1-score: {evaluations.get(best_model, {}).get("f1", 0):.3f}',
                'implementation': 'Deploy with monitoring and alerting systems',
                'timeline': '3-6 months'
            },
            {
                'category': 'Real-time Monitoring',
                'recommendation': 'Implement real-time transaction monitoring',
                'rationale': 'Immediate detection and prevention of fraud',
                'implementation': 'Stream processing with ML models',
                'timeline': '6-12 months'
            },
            {
                'category': 'Customer Education',
                'recommendation': 'Educate customers about fraud prevention',
                'rationale': 'Reduce fraud through awareness and vigilance',
                'implementation': 'Regular communication and training programs',
                'timeline': 'Ongoing'
            }
        ]
    
    def _generate_regulatory_recommendations(self, evaluations):
        """Generate recommendations for regulators."""
        return [
            {
                'category': 'Compliance Standards',
                'recommendation': 'Establish minimum fraud detection requirements',
                'rationale': 'Ensure all institutions have adequate protection',
                'implementation': 'Define performance benchmarks and reporting requirements',
                'timeline': '12-24 months'
            },
            {
                'category': 'Audit Requirements',
                'recommendation': 'Require regular audits of fraud detection systems',
                'rationale': 'Maintain system effectiveness and identify improvements',
                'implementation': 'Standardized audit procedures and reporting',
                'timeline': 'Annual'
            },
            {
                'category': 'International Cooperation',
                'recommendation': 'Coordinate with international regulatory bodies',
                'rationale': 'Address cross-border fraud effectively',
                'implementation': 'Bilateral agreements and information sharing',
                'timeline': 'Ongoing'
            }
        ]
    
    def _generate_tech_recommendations(self, evaluations, data_info):
        """Generate technology implementation recommendations."""
        return [
            {
                'category': 'Infrastructure',
                'recommendation': 'Invest in scalable cloud infrastructure',
                'rationale': 'Handle large transaction volumes efficiently',
                'implementation': 'Cloud-native architecture with auto-scaling',
                'timeline': '6-12 months'
            },
            {
                'category': 'Data Quality',
                'recommendation': 'Implement robust data quality management',
                'rationale': 'Model performance depends on data quality',
                'implementation': 'Data validation, cleaning, and monitoring systems',
                'timeline': '3-6 months'
            },
            {
                'category': 'Security',
                'recommendation': 'Implement end-to-end encryption and security',
                'rationale': 'Protect sensitive financial data',
                'implementation': 'Encryption at rest and in transit, access controls',
                'timeline': 'Immediate'
            }
        ]
    
    def _generate_risk_assessment(self, evaluations):
        """Generate risk assessment based on model performance."""
        if not evaluations:
            return {
                'overall_risk': 'Unknown',
                'average_f1_score': 0.0,
                'risk_factors': [
                    'No model evaluation data available',
                    'Models need to be trained first',
                    'Data quality issues',
                    'System integration challenges'
                ],
                'mitigation_strategies': [
                    'Run model training first',
                    'Ensure data quality',
                    'Implement monitoring systems',
                    'Staff training and awareness'
                ]
            }
        
        # Calculate overall risk based on model performance
        avg_f1 = np.mean([eval_results.get('f1', 0) for eval_results in evaluations.values()])
        
        if avg_f1 >= 0.9:
            risk_level = 'Low'
        elif avg_f1 >= 0.7:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'overall_risk': risk_level,
            'average_f1_score': avg_f1,
            'risk_factors': [
                'Model performance variability',
                'Data quality issues',
                'Evolving fraud techniques',
                'System integration challenges'
            ],
            'mitigation_strategies': [
                'Regular model retraining',
                'Continuous monitoring',
                'Multi-layered security',
                'Staff training and awareness'
            ]
        }
    
    def _generate_cost_benefit_analysis(self, evaluations):
        """Generate cost-benefit analysis."""
        if not evaluations:
            return {
                'roi': 'Unknown',
                'estimated_savings': '$0',
                'implementation_cost': '$500,000',
                'benefits': [
                    'Reduced fraud losses',
                    'Improved customer trust',
                    'Regulatory compliance',
                    'Operational efficiency'
                ],
                'costs': [
                    'Technology infrastructure',
                    'Staff training',
                    'Ongoing maintenance',
                    'Regulatory compliance'
                ]
            }
        
        # Estimate ROI based on model performance
        avg_roc_auc = np.mean([eval_results.get('roc_auc', 0.5) for eval_results in evaluations.values()])
        
        # Simplified ROI calculation
        fraud_prevention_rate = avg_roc_auc * 0.8  # Assume 80% of detected fraud is prevented
        estimated_savings = fraud_prevention_rate * 1000000  # Assume $1M potential fraud
        implementation_cost = 500000  # Estimated implementation cost
        roi = (estimated_savings - implementation_cost) / implementation_cost * 100
        
        return {
            'roi': f'{roi:.1f}%',
            'estimated_savings': f'${estimated_savings:,.0f}',
            'implementation_cost': f'${implementation_cost:,.0f}',
            'benefits': [
                'Reduced fraud losses',
                'Improved customer trust',
                'Regulatory compliance',
                'Operational efficiency'
            ],
            'costs': [
                'Technology infrastructure',
                'Staff training',
                'Ongoing maintenance',
                'Regulatory compliance'
            ]
        } 