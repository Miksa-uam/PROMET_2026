# enhanced_visualization.py
"""
Enhanced Visualization Module for Random Forest Feature Importance

This module provides publication-ready composite plots for feature importance analysis
with statistical significance annotations and professional styling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Optional, Tuple
import logging
from significance_testing import SignificanceResults

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedFeatureImportancePlotter:
    """
    Publication-ready feature importance visualization with statistical significance.
    
    This class creates composite plots that combine different feature importance metrics
    with proper statistical annotations, consistent styling, and dynamic sizing to
    accommodate all features without collapsing.
    """
    
    def __init__(self, config, nice_names: Dict[str, str] = None):
        """
        Initialize the enhanced plotter.
        
        Args:
            config: paper2_rf_config instance with visualization settings
            nice_names: Optional dictionary mapping feature names to display names
        """
        self.config = config
        self.nice_names = nice_names or {}
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Color schemes for different plot types
        self.colors = {
            'gini': '#2E86AB',
            'permutation': '#A23B72', 
            'shap_bar': '#F18F01',
            'shap_beeswarm': None,  # Use SHAP default colormap
            'significance_line': '#E63946',
            'significant_text': '#E63946',
            'non_significant_text': '#6C757D'
        }
        
        logger.info("Initialized EnhancedFeatureImportancePlotter")
    
    def _get_nice_name(self, feature_name: str) -> str:
        """Get the display name for a feature."""
        return self.nice_names.get(feature_name, feature_name)
    
    def _calculate_figure_dimensions(self, n_features: int, plot_type: str = 'primary') -> Tuple[float, float]:
        """
        Calculate optimal figure dimensions based on number of features.
        
        Args:
            n_features: Number of features to display
            plot_type: 'primary' or 'secondary' for different base dimensions
            
        Returns:
            Tuple of (width, height) in inches
        """
        if plot_type == 'primary':
            base_width = self.config.figure_width_primary
            base_height = self.config.figure_height_primary
        else:
            base_width = self.config.figure_width_secondary
            base_height = self.config.figure_height_secondary
        
        # Adjust height based on number of features to ensure readability
        min_height_per_feature = 0.4  # Minimum inches per feature
        calculated_height = max(base_height, n_features * min_height_per_feature + 2)
        
        # Ensure reasonable bounds
        max_height = 20.0
        calculated_height = min(calculated_height, max_height)
        
        logger.debug(f"Calculated dimensions for {n_features} features: {base_width}x{calculated_height}")
        return base_width, calculated_height
    
    def _determine_feature_ordering(self, importance_dict: Dict[str, float], 
                                  ascending: bool = False) -> List[str]:
        """
        Determine feature ordering based on importance values.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance values
            ascending: If True, sort in ascending order (lowest first)
            
        Returns:
            List of feature names in the specified order
        """
        if not importance_dict:
            return []
        
        # Sort by importance values
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=not ascending)
        feature_order = [feat for feat, _ in sorted_features]
        
        logger.debug(f"Feature ordering ({'ascending' if ascending else 'descending'}): {feature_order}")
        return feature_order
    
    def _add_significance_annotations(self, ax, feature_names: List[str], 
                                    significant_features: List[str], 
                                    y_positions: List[int]) -> None:
        """
        Add significance asterisks to feature labels.
        
        Args:
            ax: Matplotlib axis object
            feature_names: List of all feature names
            significant_features: List of significant feature names
            y_positions: Y-axis positions corresponding to features
        """
        try:
            for i, feature in enumerate(feature_names):
                if feature in significant_features:
                    # Add asterisk for significant features
                    y_pos = y_positions[i]
                    current_label = ax.get_yticklabels()[i].get_text()
                    
                    # Determine significance level for asterisks
                    asterisk = "*"  # Default for p < 0.05
                    # Could be enhanced to show ** for p < 0.01 if p-values are available
                    
                    # Update the label with asterisk
                    new_label = f"{current_label} {asterisk}"
                    ax.get_yticklabels()[i].set_text(new_label)
                    ax.get_yticklabels()[i].set_color(self.colors['significant_text'])
                else:
                    # Non-significant features in gray
                    ax.get_yticklabels()[i].set_color(self.colors['non_significant_text'])
                    
        except Exception as e:
            logger.warning(f"Failed to add significance annotations: {str(e)}")
    
    def _add_significance_threshold_line(self, ax, threshold: float, 
                                       x_limits: Tuple[float, float]) -> None:
        """
        Add vertical dashed line showing significance threshold.
        
        Args:
            ax: Matplotlib axis object
            threshold: Significance threshold value
            x_limits: Tuple of (min, max) x-axis limits
        """
        try:
            ax.axvline(x=threshold, color=self.colors['significance_line'], 
                      linestyle='--', linewidth=2, alpha=0.8, 
                      label=f'Significance threshold ({threshold:.4f})')
            
            # Add text annotation
            y_pos = ax.get_ylim()[1] * 0.95  # Near top of plot
            ax.text(threshold, y_pos, f'Threshold\n{threshold:.4f}', 
                   ha='center', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                   
        except Exception as e:
            logger.warning(f"Failed to add significance threshold line: {str(e)}")
    
    def _create_gini_importance_panel(self, ax, gini_importance: pd.Series, 
                                    significance_results: SignificanceResults = None) -> None:
        """
        Create Gini importance panel with significance testing.
        
        Args:
            ax: Matplotlib axis object
            gini_importance: Series with Gini importance values
            significance_results: Optional significance test results
        """
        try:
            # Determine feature ordering (descending by importance)
            feature_order = self._determine_feature_ordering(gini_importance.to_dict(), ascending=False)
            
            # Reorder data
            ordered_importance = gini_importance.reindex(feature_order)
            nice_labels = [self._get_nice_name(feat) for feat in feature_order]
            
            # Create horizontal bar plot
            y_positions = range(len(ordered_importance))
            bars = ax.barh(y_positions, ordered_importance.values, color=self.colors['gini'], alpha=0.8)
            
            # Set labels and title
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Gini Importance (Mean Decrease in Impurity)', fontsize=11)
            ax.set_title('Gini Feature Importance', fontsize=12, fontweight='bold')
            
            # Add significance annotations if available
            if significance_results and significance_results.gini_significant_features:
                self._add_significance_annotations(
                    ax, feature_order, significance_results.gini_significant_features, y_positions
                )
                
                # Add significance threshold line
                if significance_results.gini_threshold > 0:
                    x_limits = ax.get_xlim()
                    self._add_significance_threshold_line(ax, significance_results.gini_threshold, x_limits)
            
            # Improve layout
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            logger.info(f"Created Gini importance panel with {len(feature_order)} features")
            
        except Exception as e:
            logger.error(f"Failed to create Gini importance panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating Gini plot', ha='center', va='center', transform=ax.transAxes)
    
    def _create_shap_beeswarm_panel(self, ax, shap_explanation, 
                                  significance_results: SignificanceResults = None) -> None:
        """
        Create SHAP beeswarm panel with significance annotations.
        
        Args:
            ax: Matplotlib axis object
            shap_explanation: SHAP Explanation object
            significance_results: Optional significance test results
        """
        try:
            # Create SHAP beeswarm plot
            plt.sca(ax)  # Set current axis for SHAP
            
            # Determine feature ordering based on mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_explanation.values), axis=0)
            feature_importance_dict = dict(zip(shap_explanation.feature_names, mean_abs_shap))
            feature_order_indices = [shap_explanation.feature_names.index(feat) 
                                   for feat in self._determine_feature_ordering(feature_importance_dict, ascending=False)]
            
            # Create ordered explanation for plotting
            ordered_explanation = shap.Explanation(
                values=shap_explanation.values[:, feature_order_indices],
                base_values=shap_explanation.base_values,
                data=shap_explanation.data[:, feature_order_indices] if shap_explanation.data is not None else None,
                feature_names=[self._get_nice_name(shap_explanation.feature_names[i]) for i in feature_order_indices]
            )
            
            # Create beeswarm plot
            shap.plots.beeswarm(
                ordered_explanation,
                max_display=self.config.max_features_display or len(shap_explanation.feature_names),
                show=False
            )
            
            ax.set_title('SHAP Value Distribution', fontsize=12, fontweight='bold')
            
            # Add significance annotations if available
            if significance_results and significance_results.shap_significant_features:
                # Get current y-tick labels and modify them
                current_labels = [label.get_text() for label in ax.get_yticklabels()]
                original_features = [shap_explanation.feature_names[i] for i in feature_order_indices]
                
                for i, (original_feat, display_label) in enumerate(zip(original_features, current_labels)):
                    if original_feat in significance_results.shap_significant_features:
                        # Add asterisk for significant features
                        new_label = f"{display_label} *"
                        ax.get_yticklabels()[i].set_text(new_label)
                        ax.get_yticklabels()[i].set_color(self.colors['significant_text'])
                    else:
                        ax.get_yticklabels()[i].set_color(self.colors['non_significant_text'])
            
            logger.info(f"Created SHAP beeswarm panel with {len(shap_explanation.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Failed to create SHAP beeswarm panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating SHAP beeswarm plot', ha='center', va='center', transform=ax.transAxes)
    
    def _create_mean_shap_panel(self, ax, shap_explanation) -> None:
        """
        Create mean absolute SHAP values panel.
        
        Args:
            ax: Matplotlib axis object
            shap_explanation: SHAP Explanation object
        """
        try:
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_explanation.values), axis=0)
            shap_importance = pd.Series(mean_abs_shap, index=shap_explanation.feature_names)
            
            # Determine feature ordering (descending by mean absolute SHAP)
            feature_order = self._determine_feature_ordering(shap_importance.to_dict(), ascending=False)
            
            # Reorder data
            ordered_importance = shap_importance.reindex(feature_order)
            nice_labels = [self._get_nice_name(feat) for feat in feature_order]
            
            # Create horizontal bar plot
            y_positions = range(len(ordered_importance))
            bars = ax.barh(y_positions, ordered_importance.values, color=self.colors['shap_bar'], alpha=0.8)
            
            # Set labels and title
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11)
            ax.set_title('Mean Absolute SHAP Values', fontsize=12, fontweight='bold')
            
            # Improve layout
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            logger.info(f"Created mean SHAP panel with {len(feature_order)} features")
            
        except Exception as e:
            logger.error(f"Failed to create mean SHAP panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating mean SHAP plot', ha='center', va='center', transform=ax.transAxes)
    
    def _create_permutation_importance_panel(self, ax, permutation_importance: pd.Series) -> None:
        """
        Create permutation importance panel.
        
        Args:
            ax: Matplotlib axis object
            permutation_importance: Series with permutation importance values
        """
        try:
            # Determine feature ordering (descending by importance)
            feature_order = self._determine_feature_ordering(permutation_importance.to_dict(), ascending=False)
            
            # Reorder data
            ordered_importance = permutation_importance.reindex(feature_order)
            nice_labels = [self._get_nice_name(feat) for feat in feature_order]
            
            # Create horizontal bar plot
            y_positions = range(len(ordered_importance))
            bars = ax.barh(y_positions, ordered_importance.values, color=self.colors['permutation'], alpha=0.8)
            
            # Set labels and title
            ax.set_yticks(y_positions)
            ax.set_yticklabels(nice_labels, fontsize=10)
            ax.set_xlabel('Permutation Importance (Mean Decrease in Score)', fontsize=11)
            ax.set_title('Permutation Feature Importance', fontsize=12, fontweight='bold')
            
            # Improve layout
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            logger.info(f"Created permutation importance panel with {len(feature_order)} features")
            
        except Exception as e:
            logger.error(f"Failed to create permutation importance panel: {str(e)}")
            ax.text(0.5, 0.5, 'Error creating permutation plot', ha='center', va='center', transform=ax.transAxes)
    
    def plot_primary_composite(self, gini_importance: pd.Series, shap_explanation,
                             significance_results: SignificanceResults = None,
                             output_path: str = None) -> None:
        """
        Create primary composite plot: Gini importance + SHAP beeswarm.
        
        Args:
            gini_importance: Series with Gini importance values
            shap_explanation: SHAP Explanation object
            significance_results: Optional significance test results
            output_path: Optional path to save the plot
        """
        try:
            logger.info("Creating primary composite plot (Gini + SHAP beeswarm)")
            
            # Calculate figure dimensions
            n_features = len(gini_importance)
            fig_width, fig_height = self._calculate_figure_dimensions(n_features, 'primary')
            
            # Create figure with two panels
            fig, (ax_gini, ax_shap) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            fig.suptitle(f'Feature Importance Analysis: {self.config.analysis_name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Create Gini importance panel (left)
            self._create_gini_importance_panel(ax_gini, gini_importance, significance_results)
            
            # Create SHAP beeswarm panel (right)
            self._create_shap_beeswarm_panel(ax_shap, shap_explanation, significance_results)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
            
            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Primary composite plot saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create primary composite plot: {str(e)}")
            raise
    
    def plot_secondary_composite(self, shap_explanation, permutation_importance: pd.Series,
                               output_path: str = None) -> None:
        """
        Create secondary composite plot: Mean SHAP + Permutation importance.
        
        Args:
            shap_explanation: SHAP Explanation object
            permutation_importance: Series with permutation importance values
            output_path: Optional path to save the plot
        """
        try:
            logger.info("Creating secondary composite plot (Mean SHAP + Permutation)")
            
            # Calculate figure dimensions
            n_features = len(permutation_importance)
            fig_width, fig_height = self._calculate_figure_dimensions(n_features, 'secondary')
            
            # Create figure with two panels
            fig, (ax_shap, ax_perm) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            fig.suptitle(f'Additional Feature Importance Metrics: {self.config.analysis_name}', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Create mean SHAP panel (left)
            self._create_mean_shap_panel(ax_shap, shap_explanation)
            
            # Create permutation importance panel (right)
            self._create_permutation_importance_panel(ax_perm, permutation_importance)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
            
            # Save if path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
                logger.info(f"Secondary composite plot saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create secondary composite plot: {str(e)}")
            raise
    
    def create_all_plots(self, gini_importance: pd.Series, shap_explanation, 
                        permutation_importance: pd.Series, 
                        significance_results: SignificanceResults = None,
                        output_dir: str = None) -> Dict[str, str]:
        """
        Create all enhanced feature importance plots.
        
        Args:
            gini_importance: Series with Gini importance values
            shap_explanation: SHAP Explanation object
            permutation_importance: Series with permutation importance values
            significance_results: Optional significance test results
            output_dir: Optional directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths (if saved)
        """
        try:
            logger.info("Creating all enhanced feature importance plots")
            
            output_paths = {}
            
            # Generate output paths if directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                primary_path = os.path.join(output_dir, f"{self.config.analysis_name}_primary_FI_composite.png")
                secondary_path = os.path.join(output_dir, f"{self.config.analysis_name}_secondary_FI_composite.png")
            else:
                primary_path = None
                secondary_path = None
            
            # Create primary composite plot
            self.plot_primary_composite(
                gini_importance, shap_explanation, significance_results, primary_path
            )
            if primary_path:
                output_paths['primary_composite'] = primary_path
            
            # Create secondary composite plot
            self.plot_secondary_composite(
                shap_explanation, permutation_importance, secondary_path
            )
            if secondary_path:
                output_paths['secondary_composite'] = secondary_path
            
            logger.info(f"All enhanced plots created successfully. Saved: {len(output_paths)} plots")
            return output_paths
            
        except Exception as e:
            logger.error(f"Failed to create all plots: {str(e)}")
            raise