# rf_feature_importances.py
"""
Random Forest Feature Importance Analysis Module

A self-contained module for Random Forest feature importance analysis with statistical 
significance testing and publication-ready visualizations.

Usage:
    from rf_feature_importances import RandomForestAnalyzer
    from paper12_config import paper2_rf_config
    
    config = paper2_rf_config(...)
    analyzer = RandomForestAnalyzer(config)
    analyzer.run_and_generate_outputs()
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from packaging import version
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, mean_squared_error
from sklearn.inspection import permutation_importance

from paper12_config import paper2_rf_config
from fdr_correction_utils import apply_fdr_correction

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SignificanceResults:
    """Container for feature importance significance test results."""
    
    # Gini significance results
    gini_threshold: float
    gini_significant_features: List[str]
    shadow_importances: Dict[str, float]
    
    # SHAP significance results  
    shap_pvalues: Dict[str, float]
    shap_adjusted_pvalues: Dict[str, float]
    shap_significant_features: List[str]
    
    # Metadata
    alpha_level: float
    n_features_tested: int
    n_shadow_features: int


class RandomForestAnalyzer:
    """
    Comprehensive Random Forest feature importance analyzer with statistical significance testing.
    
    This class provides a complete pipeline for Random Forest analysis including:
    - Data loading and preprocessing
    - Model training with optional hyperparameter tuning
    - Feature importance calculation (Gini, Permutation, SHAP)
    - Statistical significance testing using shadow features and Wilcoxon tests
    - Publication-ready visualizations with significance annotations
    
    Key principle: Use snake_case feature names internally for all operations,
    nice names only for display in plots.
    """

    def __init__(self, config: paper2_rf_config):
        """Initialize the Random Forest analyzer."""
        self.config = config
        self.model = None
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.results = {}
        
        # Color schemes for visualizations
        self.colors = {
            'gini': '#2E86AB',
            'permutation': '#A23B72', 
            'shap_bar': '#F18F01',
            'shap_beeswarm': None,  # Use SHAP default colormap
            'significance_line': '#E63946',
            'significant_text': '#E63946',
            'non_significant_text': '#6C757D'
        }
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"--- Initialized Analysis: {self.config.analysis_name} ---")
        print(f"--- Significance testing enabled: Gini={config.enable_gini_significance}, SHAP={config.enable_shap_significance} ---")

    def _get_nice_name(self, feature_name: str) -> str:
        """Get display name for a feature. Use snake_case internally, nice names only for display."""
        return self.config.nice_names.get(feature_name, feature_name)

    def _load_and_prepare_data(self):
        """Load data and prepare the outcome variable."""
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
        """Train the model and calculate all metrics and importances."""
        self._load_and_prepare_data()
        
        # Define features (X) and target (y) - use original snake_case names
        all_predictors = self.config.predictors + self.config.covariates
        X = self.data[all_predictors]
        y = self.data['outcome']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, stratify=y if self.config.model_type == 'classifier' else None, random_state=42
        )
        
        # Initialize and train model
        if self.config.model_type == 'classifier':
            base_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
        else:
            base_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

        if self.config.run_hyperparameter_tuning:
            print(">>> Running GridSearchCV for hyperparameter tuning...")
            param_grid = {
                'max_depth': [5, 10, 20, None],
                'min_samples_leaf': [2, 4, 8],
                'min_samples_split': [5, 10, 20]
            }
            grid_search = GridSearchCV(
                estimator=base_model, param_grid=param_grid, cv=5, n_jobs=-1,
                scoring='roc_auc' if self.config.model_type == 'classifier' else 'neg_mean_squared_error',
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            print(f">>> Best params found: {grid_search.best_params_}")
        else:
            print(">>> Skipping hyperparameter tuning. Using default model.")
            self.model = base_model
            self.model.fit(self.X_train, self.y_train)

        print("Model training complete.")

        # Calculate performance metrics
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

        # Feature Importances - store with original snake_case names
        self.results['gini_importance'] = pd.Series(self.model.feature_importances_, index=all_predictors)
        perm_imp = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1)
        self.results['permutation_importance'] = pd.Series(perm_imp.importances_mean, index=all_predictors)

        # SHAP values
        print("Calculating SHAP values...")
        try:
            if self.config.model_type == 'classifier':
                background_data = self.X_train.sample(n=min(100, len(self.X_train)), random_state=42)
                explainer = shap.TreeExplainer(self.model, data=background_data, 
                                             feature_perturbation="interventional", model_output="probability")
                shap_values = explainer(self.X_test)
            else:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer(self.X_test)
        except Exception as e:
            print(f"Warning: SHAP calculation failed. Falling back to basic explanation. Error: {e}")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer(self.X_test)

        self.results['shap_explanation'] = shap_values
        print("All calculations complete.")

    def _create_shadow_features(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Create shadow features by randomly shuffling each original feature."""
        X_shadow = X_train.copy()
        np.random.seed(42)
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f"{col}_shadow" for col in X_shadow.columns]
        return X_shadow

    def _test_gini_significance(self) -> Tuple[float, List[str], Dict[str, float]]:
        """Test Gini importance significance using shadow feature permutation."""
        if not self.config.enable_gini_significance:
            return 0.0, [], {}
        
        try:
            print("Testing Gini significance with shadow features...")
            X_shadow = self._create_shadow_features(self.X_train)
            X_augmented = pd.concat([self.X_train, X_shadow], axis=1)
            
            # Train model on augmented dataset
            if self.config.model_type == 'classifier':
                model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            
            model.fit(X_augmented, self.y_train)
            
            # Extract importances
            all_importances = dict(zip(X_augmented.columns, model.feature_importances_))
            shadow_importances = {feat: all_importances[feat] for feat in X_shadow.columns}
            original_importances = {feat: all_importances[feat] for feat in self.X_train.columns}
            
            # Calculate threshold and significant features
            threshold = max(shadow_importances.values()) if shadow_importances else 0.0
            significant_features = [feat for feat, imp in original_importances.items() if imp > threshold]
            
            # DEBUG: Print detailed results
            print(f"DEBUG Gini significance: threshold={threshold:.6f}")
            print(f"DEBUG Original importances: {original_importances}")
            print(f"DEBUG Significant features: {significant_features}")
            print(f"DEBUG Features above threshold:")
            for feat, imp in original_importances.items():
                above_threshold = imp > threshold
                print(f"  {feat}: {imp:.6f} > {threshold:.6f} = {above_threshold}")
            
            return threshold, significant_features, shadow_importances
            
        except Exception as e:
            print(f"Warning: Gini significance testing failed: {str(e)}")
            return 0.0, [], {}

    def _test_shap_significance(self, shap_values: np.ndarray, feature_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        """Test SHAP value significance using Wilcoxon signed-rank tests with FDR correction."""
        if not self.config.enable_shap_significance:
            return {}, {}, []
        
        try:
            print("Testing SHAP significance with Wilcoxon tests...")
            raw_pvalues = {}
            
            for i, feature_name in enumerate(feature_names):
                feature_shap_values = shap_values[:, i]
                if np.all(feature_shap_values == 0):
                    raw_pvalues[feature_name] = 1.0
                    continue
                
                try:
                    _, p_value = wilcoxon(feature_shap_values, alternative='two-sided')
                    raw_pvalues[feature_name] = p_value
                except ValueError:
                    raw_pvalues[feature_name] = 1.0
            
            # Apply FDR correction
            p_values_list = list(raw_pvalues.values())
            adjusted_p_values_list = apply_fdr_correction(p_values_list, alpha=self.config.significance_alpha)
            adjusted_pvalues = dict(zip(feature_names, adjusted_p_values_list))
            
            # Identify significant features
            significant_features = [feat for feat, adj_p in adjusted_pvalues.items() 
                                  if adj_p < self.config.significance_alpha]
            
            print(f"SHAP significance: {len(significant_features)} significant after FDR correction")
            return raw_pvalues, adjusted_pvalues, significant_features
            
        except Exception as e:
            print(f"Warning: SHAP significance testing failed: {str(e)}")
            return {}, {}, []

    def _test_feature_significance(self):
        """Run comprehensive significance testing for both Gini and SHAP importance."""
        print("Starting feature importance significance testing...")
        
        # Get SHAP explanation for positive class if classifier
        shap_explanation = self.results['shap_explanation']
        exp = self._shap_explanation_for_positive_class(shap_explanation)
        
        # Run significance tests
        gini_threshold, gini_significant, shadow_importances = self._test_gini_significance()
        shap_pvalues, shap_adjusted_pvalues, shap_significant = self._test_shap_significance(exp.values, exp.feature_names)
        
        # Store results
        all_predictors = self.config.predictors + self.config.covariates
        significance_results = SignificanceResults(
            gini_threshold=gini_threshold,
            gini_significant_features=gini_significant,
            shadow_importances=shadow_importances,
            shap_pvalues=shap_pvalues,
            shap_adjusted_pvalues=shap_adjusted_pvalues,
            shap_significant_features=shap_significant,
            alpha_level=self.config.significance_alpha,
            n_features_tested=len(all_predictors),
            n_shadow_features=len(shadow_importances)
        )
        
        self.results['significance_results'] = significance_results
        print(f"Significance testing complete: Gini={len(gini_significant)}, SHAP={len(shap_significant)}")
        return significance_results

    def _shap_explanation_for_positive_class(self, exp):
        """Extract SHAP explanation for positive class in binary classification."""
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
    
    def _add_significance_annotations(self, ax, ordered_snake_case_names: List[str], 
                                    significant_snake_case_names: List[str]) -> None:
        """
        Add significance asterisks and red coloring to feature labels.
        
        CRITICAL FIX: This method properly handles the name matching by:
        1. Working with original snake_case names for logic (significant_snake_case_names)
        2. Modifying the display labels (nice names) that are already set on the plot
        3. Ensuring the order matches between snake_case names and plot labels
        
        Args:
            ax: Matplotlib axis object
            ordered_snake_case_names: Original feature names in plot order
            significant_snake_case_names: List of significant original feature names
        """
        try:
            # DEBUG: Print what we're working with
            print(f"DEBUG: Ordered snake_case names: {ordered_snake_case_names}")
            print(f"DEBUG: Significant snake_case names: {significant_snake_case_names}")
            
            for i, original_feature_name in enumerate(ordered_snake_case_names):
                current_label = ax.get_yticklabels()[i].get_text()
                print(f"DEBUG: Feature {i}: '{original_feature_name}' -> '{current_label}' -> Significant: {original_feature_name in significant_snake_case_names}")
                
                if original_feature_name in significant_snake_case_names:
                    # Get current display label (nice name) and add asterisk
                    new_label = f"{current_label} *"
                    
                    # Update label with asterisk and red color
                    ax.get_yticklabels()[i].set_text(new_label)
                    ax.get_yticklabels()[i].set_color(self.colors['significant_text'])
                    print(f"DEBUG: Applied significance to '{original_feature_name}': '{new_label}' in red")
                else:
                    # Non-significant features in gray
                    ax.get_yticklabels()[i].set_color(self.colors['non_significant_text'])
                    print(f"DEBUG: Applied gray to '{original_feature_name}'")
                    
        except Exception as e:
            print(f"ERROR in significance annotations: {str(e)}")
            import traceback
            traceback.print_exc()

    def _add_significance_threshold_line(self, ax, threshold: float) -> None:
        """Add vertical dashed line showing significance threshold."""
        try:
            ax.axvline(x=threshold, color=self.colors['significance_line'], 
                      linestyle='--', linewidth=2, alpha=0.8)
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95
            ax.text(threshold, y_pos, f'Threshold\\n{threshold:.4f}', 
                   ha='center', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                   
        except Exception as e:
            logger.warning(f"Failed to add significance threshold line: {str(e)}")

    def _create_gini_importance_panel(self, ax, gini_importance: pd.Series, 
                                    significance_results: SignificanceResults = None) -> None:
        """Create Gini importance panel with significance testing."""
        try:
            # Step 1: Order features by importance (descending) - use original snake_case names
            importance_items = sorted(gini_importance.items(), key=lambda x: x[1], reverse=True)
            ordered_snake_case_names = [name for name, _ in importance_items]
            ordered_values = [value for _, value in importance_items]
            
            # Step 2: Create nice labels for display ONLY
            nice_labels = [self._get_nice_name(name) for name in ordered_snake_case_names]
            
            # Step 3: Create plot with nice labels for display
            y_positions = range(len(ordered_snake_case_names))
            ax.barh(y_positions, ordered_values, color=self.colors['gini'], alpha=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Gini Importance (Mean Decrease in Impurity)', fontsize=11)
            ax.set_title('Gini Feature Importance', fontsize=12, fontweight='bold')
            
            # Step 4: Add significance annotations - use original names for logic
            print(f"DEBUG GINI: Checking significance results...")
            print(f"DEBUG GINI: significance_results exists = {significance_results is not None}")
            if significance_results:
                print(f"DEBUG GINI: gini_significant_features = {significance_results.gini_significant_features}")
                print(f"DEBUG GINI: len(gini_significant_features) = {len(significance_results.gini_significant_features)}")
                
                if significance_results.gini_significant_features:
                    print(f"DEBUG GINI: About to call _add_significance_annotations")
                    print(f"DEBUG GINI: ordered_snake_case_names = {ordered_snake_case_names}")
                    self._add_significance_annotations(
                        ax, ordered_snake_case_names, significance_results.gini_significant_features
                    )
                else:
                    print(f"DEBUG GINI: No significant features to annotate")
                
                # Add significance threshold line (always show if we have results)
                if significance_results.gini_threshold > 0:
                    print(f"DEBUG GINI: Adding threshold line at {significance_results.gini_threshold}")
                    self._add_significance_threshold_line(ax, significance_results.gini_threshold)
            else:
                print(f"DEBUG GINI: No significance results available")
            
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
        except Exception as e:
            print(f"ERROR: Failed to create Gini importance panel: {str(e)}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, 'Error creating Gini plot', ha='center', va='center', transform=ax.transAxes)

    def _create_shap_beeswarm_panel(self, ax, shap_explanation, 
                                  significance_results: SignificanceResults = None) -> None:
        """Create SHAP beeswarm panel with significance annotations."""
        try:
            plt.sca(ax)  # Set current axis for SHAP
            
            # Step 1: Order features by mean absolute SHAP values - use original names
            mean_abs_shap = np.mean(np.abs(shap_explanation.values), axis=0)
            shap_items = sorted(zip(shap_explanation.feature_names, mean_abs_shap), 
                               key=lambda x: x[1], reverse=True)
            ordered_snake_case_names = [name for name, _ in shap_items]
            feature_order_indices = [shap_explanation.feature_names.index(name) for name in ordered_snake_case_names]
            
            # Step 2: Create ordered explanation with nice names for display
            ordered_explanation = shap.Explanation(
                values=shap_explanation.values[:, feature_order_indices],
                base_values=shap_explanation.base_values,
                data=shap_explanation.data[:, feature_order_indices] if shap_explanation.data is not None else None,
                feature_names=[self._get_nice_name(name) for name in ordered_snake_case_names]
            )
            
            # Step 3: Create beeswarm plot
            shap.plots.beeswarm(ordered_explanation, 
                               max_display=self.config.max_features_display or len(shap_explanation.feature_names),
                               show=False)
            ax.set_title('SHAP Value Distribution', fontsize=12, fontweight='bold')
            
            # Step 4: Add significance annotations - use original names for logic
            if significance_results and significance_results.shap_significant_features:
                self._add_significance_annotations(
                    ax, ordered_snake_case_names, significance_results.shap_significant_features
                )
            
        except Exception as e:
            logger.error(f"Failed to create SHAP beeswarm panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating SHAP beeswarm plot', ha='center', va='center', transform=ax.transAxes)

    def _create_mean_shap_panel(self, ax, shap_explanation) -> None:
        """Create mean absolute SHAP values panel."""
        try:
            # Order by mean absolute SHAP values - use original names internally
            mean_abs_shap = np.mean(np.abs(shap_explanation.values), axis=0)
            shap_items = sorted(zip(shap_explanation.feature_names, mean_abs_shap), 
                               key=lambda x: x[1], reverse=True)
            ordered_snake_case_names = [name for name, _ in shap_items]
            ordered_values = [value for _, value in shap_items]
            
            # Create nice labels for display
            nice_labels = [self._get_nice_name(name) for name in ordered_snake_case_names]
            
            # Create plot
            y_positions = range(len(ordered_snake_case_names))
            ax.barh(y_positions, ordered_values, color=self.colors['shap_bar'], alpha=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11)
            ax.set_title('Mean Absolute SHAP Values', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Failed to create mean SHAP panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating mean SHAP plot', ha='center', va='center', transform=ax.transAxes)

    def _create_permutation_importance_panel(self, ax, permutation_importance: pd.Series) -> None:
        """Create permutation importance panel."""
        try:
            # Order by importance - use original names internally
            perm_items = sorted(permutation_importance.items(), key=lambda x: x[1], reverse=True)
            ordered_snake_case_names = [name for name, _ in perm_items]
            ordered_values = [value for _, value in perm_items]
            
            # Create nice labels for display
            nice_labels = [self._get_nice_name(name) for name in ordered_snake_case_names]
            
            # Create plot
            y_positions = range(len(ordered_snake_case_names))
            ax.barh(y_positions, ordered_values, color=self.colors['permutation'], alpha=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Permutation Importance (Mean Decrease in Score)', fontsize=11)
            ax.set_title('Permutation Feature Importance', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
        except Exception as e:
            logger.error(f"Failed to create permutation importance panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating permutation plot', ha='center', va='center', transform=ax.transAxes)

    def _calculate_figure_dimensions(self, n_features: int, plot_type: str = 'primary') -> Tuple[float, float]:
        """Calculate optimal figure dimensions based on number of features."""
        if plot_type == 'primary':
            base_width = self.config.figure_width_primary
            base_height = self.config.figure_height_primary
        else:
            base_width = self.config.figure_width_secondary
            base_height = self.config.figure_height_secondary
        
        # Adjust height based on number of features
        min_height_per_feature = 0.4
        calculated_height = max(base_height, n_features * min_height_per_feature + 2)
        calculated_height = min(calculated_height, 20.0)  # Max height limit
        
        return base_width, calculated_height

    def _plot_primary_composite(self):
        """Create primary composite plot: Gini importance + SHAP beeswarm."""
        print("DEBUG: _plot_primary_composite method called!")
        try:
            print("Generating primary composite plot (Gini + SHAP beeswarm)...")
            
            gini_importance = self.results['gini_importance']
            shap_explanation = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
            significance_results = self.results.get('significance_results')
            
            # Calculate figure dimensions
            n_features = len(gini_importance)
            fig_width, fig_height = self._calculate_figure_dimensions(n_features, 'primary')
            
            # Create figure
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax_gini, ax_shap) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            fig.suptitle(f'Feature Importance Analysis: {self.config.analysis_name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Create panels
            self._create_gini_importance_panel(ax_gini, gini_importance, significance_results)
            self._create_shap_beeswarm_panel(ax_shap, shap_explanation, significance_results)
            
            # Save and show
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            output_path = os.path.join(self.config.output_dir, f"{self.config.analysis_name}_primary_FI_composite.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Primary composite plot saved to: {output_path}")
            plt.show()
            
        except Exception as e:
            print(f"Warning: Primary composite plot generation failed: {str(e)}")

    def _plot_secondary_composite(self):
        """Create secondary composite plot: Mean SHAP + Permutation importance."""
        try:
            print("Generating secondary composite plot (Mean SHAP + Permutation)...")
            
            shap_explanation = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
            permutation_importance = self.results['permutation_importance']
            
            # Calculate figure dimensions
            n_features = len(permutation_importance)
            fig_width, fig_height = self._calculate_figure_dimensions(n_features, 'secondary')
            
            # Create figure
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, (ax_shap, ax_perm) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            fig.suptitle(f'Additional Feature Importance Metrics: {self.config.analysis_name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Create panels
            self._create_mean_shap_panel(ax_shap, shap_explanation)
            self._create_permutation_importance_panel(ax_perm, permutation_importance)
            
            # Save and show
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            output_path = os.path.join(self.config.output_dir, f"{self.config.analysis_name}_secondary_FI_composite.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Secondary composite plot saved to: {output_path}")
            plt.show()
            
        except Exception as e:
            print(f"Warning: Secondary composite plot generation failed: {str(e)}")

    def plot_roc_curve(self):
        """Plot the ROC curve for classifier models."""
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

    def run_and_generate_outputs(self):
        """
        Complete pipeline: run analysis, test significance, and generate all outputs.
        
        This is the main method that should be called from notebooks.
        """
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
        
        # Print significance summary
        if 'significance_results' in self.results:
            sig_results = self.results['significance_results']
            print(f"--- Significance Testing Summary ---")
            print(f"Gini significant features ({len(sig_results.gini_significant_features)}): {sig_results.gini_significant_features}")
            print(f"SHAP significant features ({len(sig_results.shap_significant_features)}): {sig_results.shap_significant_features}")
            print(f"Significance threshold: {sig_results.gini_threshold:.6f}")
            print(f"Alpha level: {sig_results.alpha_level}")

    def run_and_generate_outputs_legacy(self):
        """Legacy method for backward compatibility."""
        print("Running legacy analysis mode...")
        self.run_analysis()
        
        if self.config.model_type == 'classifier':
            self.plot_roc_curve()
        
        # Generate basic SHAP summary plot
        exp = self._shap_explanation_for_positive_class(self.results['shap_explanation'])
        shap.plots.beeswarm(exp, show=False)
        plt.title(f"{self.config.analysis_name}: SHAP Summary")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.analysis_name}_shap_summary.png"), dpi=300)
        plt.show()
        
        print(f"--- Legacy Analysis Complete: {self.config.analysis_name} ---")