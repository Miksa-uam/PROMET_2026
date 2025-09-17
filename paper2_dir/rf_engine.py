# In a new file: rf_engine.py

import os
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from packaging import version

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
        
        # Initialize a base model instance
        if self.config.model_type == 'classifier':
            base_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        else: # regressor
            base_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

        # Conditionally run hyperparameter tuning
        if self.config.run_hyperparameter_tuning:
            print(">>> Running GridSearchCV for hyperparameter tuning...")
            
            # Define a focused parameter grid to search
            param_grid = {
                'max_depth': [5, 10, 20, None],
                'min_samples_leaf': [2, 4, 8],
                'min_samples_split': [5, 10, 20]
            }
            
            # Set up the grid search with 5-fold cross-validation
            # It uses all available CPU cores (n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring='roc_auc' if self.config.model_type == 'classifier' else 'neg_mean_squared_error',
                verbose=1 # Set to 1 to see progress
            )
            
            # Fit the grid search to find the best model
            grid_search.fit(self.X_train, self.y_train)
            
            # Set the class model to the best one found
            self.model = grid_search.best_estimator_
            print(f">>> Best params found: {grid_search.best_params_}")
            
        else:
            # If not tuning, just fit the base model as before
            print(">>> Skipping hyperparameter tuning. Using default model.")
            self.model = base_model
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

        # --- SHAP values (as Explanation) ---
        print("Calculating SHAP values...")
        shap_values = None
        try:
            if self.config.model_type == 'classifier':
                # For classifiers, we use the "interventional" method to get probability-based SHAP values.
                # This requires a background dataset, which we sample from the training data.
                background_data = self.X_train.sample(n=min(100, len(self.X_train)), random_state=42)
                
                explainer = shap.TreeExplainer(
                    self.model,
                    data=background_data,
                    feature_perturbation="interventional",
                    model_output="probability"
                )
                shap_values = explainer(self.X_test)
                
            else: # For regressor models
                # For regressors, the default explainer (raw output) is sufficient.
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer(self.X_test)

        except Exception as e:
            print(f"Warning: SHAP calculation failed. Falling back to basic explanation. Error: {e}")
            # A safe fallback that works for both model types but may not give probability-scaled values
            # for classifiers, potentially affecting plots.
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(self.X_test)

        self.results['shap_explanation'] = shap_values
        print("All calculations complete.")

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
        # plt.savefig(os.path.join(self.config.output_dir, file_name), dpi=300)
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
        # plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_roc_curve.png"), dpi=300)
        plt.show()

    def _shap_explanation_for_positive_class(self, exp):
        vals = np.array(exp.values)
        if self.config.model_type == 'classifier' and vals.ndim == 3:
            cls_idx = list(self.model.classes_).index(1)
            return shap.Explanation(
                values=exp.values[..., cls_idx],
                base_values=np.array(exp.base_values)[..., cls_idx],
                data=exp.data,
                feature_names=exp.feature_names
            )
        return exp

    def plot_shap_summary(self):
        exp = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
        shap.plots.beeswarm(exp, show=False)
        plt.title(f"{self.config.analysis_name}: SHAP Summary")
        plt.tight_layout()
        # plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_summary.png"), dpi=300)
        plt.show()

    def plot_feature_importance_grid(self):
        """
        Publication-ready 2x2 grid:
        1) Gini, 2) Permutation, 3) SHAP |mean| bar, 4) SHAP beeswarm.
        Non-uniform grid to give beeswarm more width and SHAP row more height.
        """
        print("Generating 2x2 feature importance grid...")
        if version.parse(shap.__version__) < version.parse("0.41.0"):
            raise RuntimeError(f"SHAP {shap.__version__} detected. Upgrade to >= 0.41 for shap.plots.* API.")

        # Canvas: rectangular, not square
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(24, 16), layout="constrained")

        # Give the right column more width (for beeswarm + color bar)
        # and the bottom row more height (for SHAP plots)
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1.0, 1.5],
            height_ratios=[1.0, 1.2]
        )

        ax_gini      = fig.add_subplot(gs[0, 0])
        ax_perm      = fig.add_subplot(gs[0, 1])
        ax_shap_bar  = fig.add_subplot(gs[1, 0])
        ax_beeswarm  = fig.add_subplot(gs[1, 1])

        # So constrained_layout leaves sensible gaps
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.10)

        # Title
        fig.suptitle(f"Feature importances: {self.config.analysis_name}",
                    fontsize=13, fontweight='bold')

        # --- Canonical sort order from SHAP |mean| ---
        exp = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
        shap_feature_order = np.argsort(np.mean(np.abs(exp.values), axis=0))
        feature_names_array = np.array(exp.feature_names)
        ordered_feature_names = feature_names_array[shap_feature_order]
        ordered_nice_names = [self.config.nice_names.get(f, f) for f in ordered_feature_names]

        # --- Top row: Gini and Permutation (sorted by SHAP order) ---
        gini_fi = self.results['gini_importance'].reindex(ordered_feature_names)
        ax_gini.barh(range(len(gini_fi)), gini_fi.values, color="#4C78A8")
        ax_gini.set_yticks(range(len(gini_fi)))
        ax_gini.set_yticklabels(ordered_nice_names, fontsize=10)
        ax_gini.set_title("Gini Importance", fontsize=12)
        ax_gini.set_xlabel("Mean Decrease in Impurity", fontsize=10)

        perm_fi = self.results['permutation_importance'].reindex(ordered_feature_names)
        ax_perm.barh(range(len(perm_fi)), perm_fi.values, color="#4C78A8")
        ax_perm.set_yticks(range(len(perm_fi)))
        ax_perm.set_yticklabels(ordered_nice_names, fontsize=10)
        ax_perm.set_title("Permutation Importance", fontsize=12)
        ax_perm.set_xlabel("Mean Decrease in Model Score", fontsize=10)

        # --- Bottom row: SHAP plots, with nice names and unified order ---
        exp_for_plotting = shap.Explanation(
            values=exp.values,
            base_values=exp.base_values,
            data=exp.data,
            feature_names=[self.config.nice_names.get(f, f) for f in exp.feature_names]
        )

        # SHAP |mean| bar (limit bars so labels stay readable)
        plt.sca(ax_shap_bar)
        shap.plots.bar(
            exp_for_plotting[:, shap_feature_order],
            max_display=min(12, exp.values.shape[1]),
            show=False
        )
        ax_shap_bar.set_title("SHAP Mean Absolute Value", fontsize=12)
        ax_shap_bar.tick_params(axis='y', labelsize=10)
        ax_shap_bar.tick_params(axis='x', labelsize=10)
        ax_shap_bar.set_xlabel(ax_shap_bar.get_xlabel(), fontsize=10)

        # SHAP beeswarm (slightly smaller markers; keep color bar; wider cell handles it)
        plt.sca(ax_beeswarm)
        shap.plots.beeswarm(
            exp_for_plotting[:, shap_feature_order],
            max_display=min(12, exp.values.shape[1]),
            s=12,
            color_bar=True,
            show=False
        )
        ax_beeswarm.set_title("SHAP Value Distribution (Beeswarm)", fontsize=10)
        ax_beeswarm.tick_params(axis='y', labelsize=10)
        ax_beeswarm.tick_params(axis='x', labelsize=10)
        ax_beeswarm.set_xlabel(ax_beeswarm.get_xlabel(), fontsize=10)

        out_path = os.path.join(self.config.output_dir, f"{self.config.analysis_name}_FI_Grid.png")
        # plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_dependence(self, feature_name, interaction_feature=None):
        exp = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
        # Modern API equivalent of dependence plot is scatter:
        plt.figure(figsize=(8,6))
        shap.plots.scatter(exp[:, feature_name], color=exp[:, interaction_feature] if interaction_feature else None, show=False)

        plt.title(f"{self.config.analysis_name}: SHAP Dependence for {feature_name}")
        plt.tight_layout()
        # plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_dependence_{feature_name}.png"), dpi=300)
        plt.show()

    def run_and_generate_outputs(self):
        """A convenience method to run the full pipeline and save all plots."""
        self.run_analysis()
        
        # if the model is a binary classifier, plot ROC curve
        # in regressors, other evaluation metrics like RMSE are used
        if self.config.model_type == 'classifier':
            self.plot_roc_curve()
        
        self.plot_shap_summary()
        self.plot_feature_importance_grid()
        
        print(f"--- Analysis Complete: {self.config.analysis_name} ---")
        print(f"All outputs saved to: {self.config.output_dir}")

