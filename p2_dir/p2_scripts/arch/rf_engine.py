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
from significance_testing import FeatureImportanceSignificanceTester, SignificanceResults
from enhanced_visualization import EnhancedFeatureImportancePlotter


class RandomForestAnalyzer:
    """A reusable engine for running RF feature importance analyses."""

    def __init__(self, config: paper2_rf_config):
        self.config = config
        self.model = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = {}
        
        # Initialize significance tester and enhanced plotter
        self.significance_tester = FeatureImportanceSignificanceTester(config, random_state=42)
        self.enhanced_plotter = EnhancedFeatureImportancePlotter(config, config.nice_names)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"--- Initialized Analysis: {self.config.analysis_name} ---")
        print(f"--- Significance testing enabled: Gini={config.enable_gini_significance}, SHAP={config.enable_shap_significance} ---")

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

    def _test_feature_significance(self):
        """
        Orchestrate significance testing for both Gini and SHAP importance.
        
        This method runs comprehensive significance testing and stores results
        in the results dictionary for use by visualization methods.
        """
        try:
            print("Starting feature importance significance testing...")
            
            # Get required data for significance testing
            all_predictors = self.config.predictors + self.config.covariates
            shap_explanation = self.results['shap_explanation']
            
            # Extract SHAP values for positive class if classifier
            exp = self._shap_explanation_for_positive_class(shap_explanation)
            
            # Run comprehensive significance testing
            significance_results = self.significance_tester.run_all_tests(
                self.X_train, 
                self.y_train, 
                exp.values, 
                exp.feature_names
            )
            
            # Store results
            self.results['significance_results'] = significance_results
            self.results['feature_ordering'] = all_predictors  # Store canonical ordering
            
            print(f"Significance testing complete:")
            print(f"  - Gini significant features: {len(significance_results.gini_significant_features)}")
            print(f"  - SHAP significant features: {len(significance_results.shap_significant_features)}")
            print(f"  - Significance threshold: {significance_results.gini_threshold:.6f}")
            
            return significance_results
            
        except Exception as e:
            print(f"Warning: Significance testing failed: {str(e)}")
            # Create minimal results to allow analysis to continue
            all_predictors = self.config.predictors + self.config.covariates
            minimal_results = SignificanceResults(
                gini_threshold=0.0,
                gini_significant_features=[],
                shadow_importances={},
                shap_pvalues={},
                shap_adjusted_pvalues={},
                shap_significant_features=[],
                alpha_level=self.config.significance_alpha,
                n_features_tested=len(all_predictors),
                n_shadow_features=0
            )
            self.results['significance_results'] = minimal_results
            self.results['feature_ordering'] = all_predictors
            return minimal_results

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
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_summary.png"), dpi=300)
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
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_shap_dependence(self, feature_name, interaction_feature=None):
        exp = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
        # Modern API equivalent of dependence plot is scatter:
        plt.figure(figsize=(8,6))
        shap.plots.scatter(exp[:, feature_name], color=exp[:, interaction_feature] if interaction_feature else None, show=False)

        plt.title(f"{self.config.analysis_name}: SHAP Dependence for {feature_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_dependence_{feature_name}.png"), dpi=300)
        plt.show()

    def _plot_primary_composite(self):
        """Create primary composite plot using enhanced visualization."""
        try:
            print("Generating primary composite plot (Gini + SHAP beeswarm)...")
            
            # Get required data
            gini_importance = self.results['gini_importance']
            shap_explanation = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
            significance_results = self.results.get('significance_results')
            
            # Generate output path
            output_path = os.path.join(self.config.output_dir, f"{self.config.analysis_name}_primary_FI_composite.png")
            
            # Create plot
            self.enhanced_plotter.plot_primary_composite(
                gini_importance, 
                shap_explanation, 
                significance_results, 
                output_path
            )
            
            print(f"Primary composite plot saved to: {output_path}")
            
        except Exception as e:
            print(f"Warning: Primary composite plot generation failed: {str(e)}")
            # Fall back to individual plots if composite fails
            self._plot_importance(self.results['gini_importance'], "Gini Importance", f"{self.config.analysis_name}_gini_importance.png")

    def _plot_secondary_composite(self):
        """Create secondary composite plot using enhanced visualization."""
        try:
            print("Generating secondary composite plot (Mean SHAP + Permutation)...")
            
            # Get required data
            shap_explanation = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
            permutation_importance = self.results['permutation_importance']
            
            # Generate output path
            output_path = os.path.join(self.config.output_dir, f"{self.config.analysis_name}_secondary_FI_composite.png")
            
            # Create plot
            self.enhanced_plotter.plot_secondary_composite(
                shap_explanation, 
                permutation_importance, 
                output_path
            )
            
            print(f"Secondary composite plot saved to: {output_path}")
            
        except Exception as e:
            print(f"Warning: Secondary composite plot generation failed: {str(e)}")
            # Fall back to individual plots if composite fails
            self._plot_importance(self.results['permutation_importance'], "Permutation Importance", f"{self.config.analysis_name}_permutation_importance.png")

    def run_and_generate_outputs(self):
        """A convenience method to run the full pipeline and save all plots."""
        # Run core analysis
        self.run_analysis()
        
        # Run significance testing
        self._test_feature_significance()
        
        # Generate plots
        print("Generating enhanced visualization outputs...")
        
        # ROC curve for classifiers
        if self.config.model_type == 'classifier':
            self.plot_roc_curve()
        
        # Enhanced composite plots
        self._plot_primary_composite()
        self._plot_secondary_composite()
        
        print(f"--- Analysis Complete: {self.config.analysis_name} ---")
        print(f"All outputs saved to: {self.config.output_dir}")
        
        # Print summary of significant features
        if 'significance_results' in self.results:
            sig_results = self.results['significance_results']
            print(f"--- Significance Testing Summary ---")
            print(f"Gini significant features ({len(sig_results.gini_significant_features)}): {sig_results.gini_significant_features}")
            print(f"SHAP significant features ({len(sig_results.shap_significant_features)}): {sig_results.shap_significant_features}")
            print(f"Significance threshold: {sig_results.gini_threshold:.6f}")
            print(f"Alpha level: {sig_results.alpha_level}")

    def run_and_generate_outputs_legacy(self):
        """Legacy method for backward compatibility - generates old-style plots."""
        self.run_analysis()
        
        if self.config.model_type == 'classifier':
            self.plot_roc_curve()
        
        self.plot_shap_summary()
        self.plot_feature_importance_grid()
        
        print(f"--- Legacy Analysis Complete: {self.config.analysis_name} ---")
        print(f"All outputs saved to: {self.config.output_dir}")

