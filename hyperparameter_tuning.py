import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

class HyperparameterTuner:
    def __init__(self, X, y, random_state=42):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.best_params = {}
        self.best_scores = {}

    def tune_isolation_forest(self):
        print("🔍 Tuning Isolation Forest...")

        param_grid = {
            'contamination': [0.05, 0.1, 0.2]
        }

        def anomaly_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            return f1_score(y, y_pred_binary, zero_division="warn")

        iso_forest = IsolationForest(random_state=self.random_state)
        grid_search = GridSearchCV(
            iso_forest,
            param_grid,
            scoring=anomaly_scorer,
            cv=2,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X, self.y)
        self.best_params['isolation_forest'] = grid_search.best_params_
        self.best_scores['isolation_forest'] = grid_search.best_score_

        print(f"✅ Best Isolation Forest params: {grid_search.best_params_}")
        print(f"✅ Best F1 Score: {grid_search.best_score_:.4f}")
        return grid_search.best_params_, grid_search.best_score_

    def tune_one_class_svm(self):
        print("🔍 Tuning One-Class SVM...")

        param_grid = {
            'nu': [0.05, 0.1, 0.2],
            'kernel': ['rbf', 'sigmoid']
        }

        def svm_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            return f1_score(y, y_pred_binary, zero_division="warn")

        svm = OneClassSVM()
        grid_search = GridSearchCV(
            svm,
            param_grid,
            scoring=svm_scorer,
            cv=2,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X, self.y)
        self.best_params['one_class_svm'] = grid_search.best_params_
        self.best_scores['one_class_svm'] = grid_search.best_score_

        print(f"✅ Best One-Class SVM params: {grid_search.best_params_}")
        print(f"✅ Best F1 Score: {grid_search.best_score_:.4f}")
        return grid_search.best_params_, grid_search.best_score_

    def tune_autoencoder(self):
        print("🔍 Tuning Autoencoder...")

        param_combinations = [
            {'encoding_dim': 8, 'batch_size': 128, 'epochs': 15},
            {'encoding_dim': 14, 'batch_size': 256, 'epochs': 10},
            {'encoding_dim': 10, 'batch_size': 128, 'epochs': 20}
        ]

        best_score = 0
        best_params = None

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )

        for params in param_combinations:
            try:
                autoencoder = TunedAutoencoder(
                    input_dim=self.X.shape[1],
                    encoding_dim=params['encoding_dim'],
                    learning_rate=0.001,  # Fixed learning rate
                    random_state=self.random_state
                )

                autoencoder.fit(X_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                preds, errors = autoencoder.predict(X_val)
                score = f1_score(y_val, preds, zero_division="warn")

                if score > best_score:
                    best_score = score
                    best_params = params

                print(f"Params: {params} -> F1: {score:.4f}")
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        self.best_params['autoencoder'] = best_params
        self.best_scores['autoencoder'] = best_score

        print(f"✅ Best Autoencoder params: {best_params}")
        print(f"✅ Best F1 Score: {best_score:.4f}")
        return best_params, best_score

    def tune_deep_autoencoder(self):
        print("🔍 Tuning Deep Autoencoder...")
        param_combinations = [
            {'encoding_dim': 16, 'batch_size': 128, 'epochs': 20},
            {'encoding_dim': 32, 'batch_size': 256, 'epochs': 25},
            {'encoding_dim': 24, 'batch_size': 128, 'epochs': 30}
        ]
        best_score = 0
        best_params = None
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        for params in param_combinations:
            try:
                deep_autoencoder = TunedAutoencoder(
                    input_dim=self.X.shape[1],
                    encoding_dim=params['encoding_dim'],
                    learning_rate=0.001,  # Fixed learning rate
                    random_state=self.random_state
                )
                deep_autoencoder.fit(X_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                preds, errors = deep_autoencoder.predict(X_val)
                score = f1_score(y_val, preds, zero_division="warn")
                if score > best_score:
                    best_score = score
                    best_params = params
                print(f"Params: {params} -> F1: {score:.4f}")
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        self.best_params['deep_autoencoder'] = best_params
        self.best_scores['deep_autoencoder'] = best_score
        print(f"✅ Best Deep Autoencoder params: {best_params}")
        print(f"✅ Best F1 Score: {best_score:.4f}")
        return best_params, best_score

    def tune_random_forest(self):
        print("🔍 Tuning Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
        }
        rf = RandomForestClassifier(random_state=self.random_state)
        search = RandomizedSearchCV(rf, param_grid, n_iter=5, scoring='f1', cv=3, random_state=self.random_state, n_jobs=-1, verbose=1)
        search.fit(self.X, self.y)
        self.best_params['random_forest'] = search.best_params_
        self.best_scores['random_forest'] = search.best_score_
        print(f"✅ Best Random Forest params: {search.best_params_}")
        print(f"✅ Best F1 Score: {search.best_score_:.4f}")
        return search.best_params_, search.best_score_

    def tune_xgboost(self):
        print("🔍 Tuning XGBoost...")
        param_grid = {
            'n_estimators': [50, 100, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state)
        search = RandomizedSearchCV(xgb, param_grid, n_iter=5, scoring='f1', cv=3, random_state=self.random_state, n_jobs=-1, verbose=1)
        search.fit(self.X, self.y)
        self.best_params['xgboost'] = search.best_params_
        self.best_scores['xgboost'] = search.best_score_
        print(f"✅ Best XGBoost params: {search.best_params_}")
        print(f"✅ Best F1 Score: {search.best_score_:.4f}")
        return search.best_params_, search.best_score_

    def tune_all_models(self):
        print("🚀 Starting hyperparameter tuning for all models...")
        results = {}
        errors = {}
        try:
            print("\n--- Tuning Isolation Forest ---")
            iso_params, iso_score = self.tune_isolation_forest()
            results['isolation_forest'] = {'params': iso_params, 'score': iso_score}
            print("✅ Isolation Forest tuning complete.")
        except Exception as e:
            print(f"❌ Isolation Forest tuning failed: {e}")
            errors['isolation_forest'] = str(e)
        try:
            print("\n--- Tuning One-Class SVM ---")
            svm_params, svm_score = self.tune_one_class_svm()
            results['one_class_svm'] = {'params': svm_params, 'score': svm_score}
            print("✅ One-Class SVM tuning complete.")
        except Exception as e:
            print(f"❌ One-Class SVM tuning failed: {e}")
            errors['one_class_svm'] = str(e)
        try:
            print("\n--- Tuning Autoencoder ---")
            auto_params, auto_score = self.tune_autoencoder()
            results['autoencoder'] = {'params': auto_params, 'score': auto_score}
            print("✅ Autoencoder tuning complete.")
        except Exception as e:
            print(f"❌ Autoencoder tuning failed: {e}")
            errors['autoencoder'] = str(e)
        try:
            print("\n--- Tuning Deep Autoencoder ---")
            deep_auto_params, deep_auto_score = self.tune_deep_autoencoder()
            results['deep_autoencoder'] = {'params': deep_auto_params, 'score': deep_auto_score}
            print("✅ Deep Autoencoder tuning complete.")
        except Exception as e:
            print(f"❌ Deep Autoencoder tuning failed: {e}")
            errors['deep_autoencoder'] = str(e)
        try:
            print("\n--- Tuning Random Forest ---")
            rf_params, rf_score = self.tune_random_forest()
            results['random_forest'] = {'params': rf_params, 'score': rf_score}
            print("✅ Random Forest tuning complete.")
        except Exception as e:
            print(f"❌ Random Forest tuning failed: {e}")
            errors['random_forest'] = str(e)
        try:
            print("\n--- Tuning XGBoost ---")
            xgb_params, xgb_score = self.tune_xgboost()
            results['xgboost'] = {'params': xgb_params, 'score': xgb_score}
            print("✅ XGBoost tuning complete.")
        except Exception as e:
            print(f"❌ XGBoost tuning failed: {e}")
            errors['xgboost'] = str(e)
        print("\n" + "=" * 50)
        print("🎯 HYPERPARAMETER TUNING SUMMARY")
        print("=" * 50)
        for model, result in results.items():
            print(f"\n{model.upper().replace('_', ' ')}:")
            print(f"  Best Score: {result['score']:.4f}")
            print(f"  Best Params: {result['params']}")
        if errors:
            print("\n--- ERRORS DURING TUNING ---")
            for model, err in errors.items():
                print(f"{model}: {err}")
        if results:
            best_model = max(results.keys(), key=lambda x: results[x]['score'])
            print(f"\n🏆 Best Overall Model: {best_model}")
            print(f"🏆 Best Score: {results[best_model]['score']:.4f}")
        # Return errors in the results for API visibility
        return {'results': results, 'errors': errors}

class TunedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=14, learning_rate=0.001, random_state=42):
        super(TunedAutoencoder, self).__init__()
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim, int(encoding_dim / 2)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(int(encoding_dim / 2), encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoding_dim, input_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, epochs=20, batch_size=256, verbose=1):
        if hasattr(X, 'values'):
            X = X.values
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose and epoch % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}')

    def predict(self, X, threshold=None):
        self.eval()
        with torch.no_grad():
            if hasattr(X, 'values'):
                X = X.values
            X_tensor = torch.FloatTensor(X).to(self.device)
            recon = self(X_tensor)
            recon = recon.cpu().numpy()
            errors = np.mean(np.square(X - recon), axis=1)

            if threshold is None:
                threshold = np.percentile(errors, 95)
            preds = (errors > threshold).astype(int)
            return preds, errors

def run_hyperparameter_tuning(X, y, random_state=42):
    tuner = HyperparameterTuner(X, y, random_state)
    results = tuner.tune_all_models()
    return results


