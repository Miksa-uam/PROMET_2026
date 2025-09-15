# In a new file: rf_engine.py

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, mean_squared_error
from sklearn.inspection import permutation_importance

from paper12_config import paper2_rf_config

class RandomForestAnalyzer:
    """A reusable engine for running RF feature importance analyses."""

    def __init__(self, config: paper2_rf_config):
        self.config = config
        self.model = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"--- Initialized Analysis: {self.config.analysis_name} ---")

    def _load_and_prepare_data(self):
        """Loads data and prepares the outcome variable."""
        with sqlite3.connect(self.config.db_path) as conn:
            self.data = pd.read_sql_query(f"SELECT * FROM {self.config.input_table}", conn)

        df = self.data.copy()
        
        # Prepare outcome variable
        if self.config.model_type == 'classifier':
            if self.config.threshold_direction == 'less_than_or_equal':
                df['outcome'] = (df[self.config.outcome_variable] <= self.config.classifier_threshold).astype(int)
            else: # greater_than_or_equal
                df['outcome'] = (df[self.config.outcome_variable] >= self.config.classifier_threshold).astype(int)
        else: # regressor
            df['outcome'] = df[self.config.outcome_variable]
        
        # Handle missing data
        all_predictors = self.config.predictors + self.config.covariates
        df = df.dropna(subset=all_predictors + ['outcome'])
        
        self.data = df
        print(f"Data prepared. Final sample size: {len(self.data)}")

    def run_analysis(self):
        """Trains the model and calculates all metrics and importances."""
        self._load_and_prepare_data()
        
        # Define features (X) and target (y)
        all_predictors = self.config.predictors + self.config.covariates
        X = self.data[all_predictors]
        y = self.data['outcome']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y if self.config.model_type == 'classifier' else None, random_state=42
        )

        # Initialize and train model
        if self.config.model_type == 'classifier':
            self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        else: # regressor
            self.model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        
        self.model.fit(self.X_train, self.y_train)
        print("Model training complete.")

        # --- Calculate and Store Results ---
        # Performance Metrics
        y_pred = self.model.predict(self.X_test)
        if self.config.model_type == 'classifier':
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            self.results['f1_score'] = f1_score(self.y_test, y_pred)
            self.results['auroc'] = roc_auc_score(self.y_test, y_proba)
            self.results['roc_curve'] = roc_curve(self.y_test, y_proba)
            print(f"Performance: F1={self.results['f1_score']:.3f}, AUROC={self.results['auroc']:.3f}")
        else:
            self.results['rmse'] = np.sqrt(mean_squared_error(self.y_test, y_pred))
            print(f"Performance: RMSE={self.results['rmse']:.3f}")

        # Feature Importances
        self.results['gini_importance'] = pd.Series(self.model.feature_importances_, index=all_predictors)
        perm_imp = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1)
        self.results['permutation_importance'] = pd.Series(perm_imp.importances_mean, index=all_predictors)
        
        # SHAP Values
        explainer = shap.TreeExplainer(self.model)
        self.results['shap_values'] = explainer(self.X_test)
        print("All calculations complete.")

    def _plot_importance(self, importance_series, title, file_name):
        """Generic plotting function for Gini and Permutation importance."""
        fi = importance_series.sort_values(ascending=True)
        nice_labels = [self.config.nice_names.get(feature, feature) for feature in fi.index]
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(fi)), fi.values, align='center')
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(nice_labels)
        ax.set_xlabel(title)
        ax.set_title(f"{self.config.analysis_name}: {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, file_name), dpi=300)
        plt.show()

    def plot_roc_curve(self):
        """Plots the ROC curve for classifier models."""
        if self.config.model_type != 'classifier':
            print("ROC curve is only for classifier models.")
            return

        fpr, tpr, _ = self.results['roc_curve']
        auroc = self.results['auroc']
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f"{self.config.analysis_name}: ROC Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_roc_curve.png"), dpi=300)
        plt.show()

    def plot_shap_summary(self):
        """Plots the SHAP beeswarm summary plot."""
        shap.summary_plot(self.results['shap_values'], self.X_test, show=False)
        plt.title(f"{self.config.analysis_name}: SHAP Summary")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_summary.png"), dpi=300)
        plt.show()

    def run_and_generate_outputs(self):
        """A convenience method to run the full pipeline and save all plots."""
        self.run_analysis()
        
        if self.config.model_type == 'classifier':
            self.plot_roc_curve()
        
        self._plot_importance(self.results['gini_importance'], 'Gini Feature Importance', f"{self.config.analysis_name}_gini_importance.png")
        self._plot_importance(self.results['permutation_importance'], 'Permutation Feature Importance', f"{self.config.analysis_name}_perm_importance.png")
        
        self.plot_shap_summary()
        
        print(f"--- Analysis Complete: {self.config.analysis_name} ---")
        print(f"All outputs saved to: {self.config.output_dir}")

