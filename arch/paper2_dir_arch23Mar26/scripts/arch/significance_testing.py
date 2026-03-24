# significance_testing.py
"""
Statistical Significance Testing Module for Random Forest Feature Importance

This module implements robust statistical significance testing for both Gini importance
and SHAP values using shadow feature permutation and Wilcoxon signed-rank tests respectively.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap

# Import existing FDR correction utilities
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


class FeatureImportanceSignificanceTester:
    """
    A comprehensive statistical significance testing framework for Random Forest feature importance.
    
    This class implements two complementary approaches:
    1. Shadow feature permutation testing for Gini importance (data-driven threshold)
    2. Wilcoxon signed-rank testing for SHAP values (with FDR correction)
    
    The class is designed to integrate seamlessly with the existing RandomForestAnalyzer
    while providing robust statistical rigor for feature importance interpretation.
    """
    
    def __init__(self, config, random_state: int = 42):
        """
        Initialize the significance tester.
        
        Args:
            config: paper2_rf_config instance with testing configuration
            random_state: Random seed for reproducible results
        """
        self.config = config
        self.random_state = random_state
        self.results = None
        
        # Validate configuration
        if not hasattr(config, 'enable_gini_significance'):
            raise ValueError("Configuration must have enable_gini_significance attribute")
        if not hasattr(config, 'enable_shap_significance'):
            raise ValueError("Configuration must have enable_shap_significance attribute")
        if not hasattr(config, 'significance_alpha'):
            raise ValueError("Configuration must have significance_alpha attribute")
            
        logger.info(f"Initialized FeatureImportanceSignificanceTester with alpha={config.significance_alpha}")
    
    def create_shadow_features(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Create shadow features by randomly shuffling each original feature.
        
        This method implements the shadow feature approach where each original feature
        is duplicated and its values are randomly permuted to break the relationship
        with the target while preserving the feature's distribution.
        
        Args:
            X_train: Training feature matrix
            
        Returns:
            DataFrame with shadow features (renamed with '_shadow' suffix)
        """
        try:
            logger.info(f"Creating shadow features for {len(X_train.columns)} original features")
            
            # Create a copy of the training data
            X_shadow = X_train.copy()
            
            # Shuffle each column independently
            np.random.seed(self.random_state)
            for col in X_shadow.columns:
                X_shadow[col] = np.random.permutation(X_shadow[col].values)
            
            # Rename columns to avoid confusion
            X_shadow.columns = [f"{col}_shadow" for col in X_shadow.columns]
            
            logger.info(f"Successfully created {len(X_shadow.columns)} shadow features")
            return X_shadow
            
        except Exception as e:
            logger.error(f"Failed to create shadow features: {str(e)}")
            raise
    
    def calculate_significance_threshold(self, shadow_importances: Dict[str, float]) -> float:
        """
        Calculate the significance threshold from shadow feature importances.
        
        The threshold is set as the maximum importance score achieved by any shadow feature,
        representing the highest level of importance that can be attributed to pure noise.
        
        Args:
            shadow_importances: Dictionary mapping shadow feature names to importance scores
            
        Returns:
            Significance threshold value
        """
        try:
            if not shadow_importances:
                logger.warning("No shadow importances provided, using threshold of 0")
                return 0.0
            
            threshold = max(shadow_importances.values())
            logger.info(f"Calculated significance threshold: {threshold:.6f}")
            return threshold
            
        except Exception as e:
            logger.error(f"Failed to calculate significance threshold: {str(e)}")
            return 0.0
    
    def test_gini_significance(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             model_params: Dict = None) -> Tuple[float, List[str], Dict[str, float]]:
        """
        Test Gini importance significance using shadow feature permutation.
        
        This method implements the robust shadow feature approach:
        1. Create shadow features by shuffling original features
        2. Train model on augmented dataset (original + shadow)
        3. Use maximum shadow importance as significance threshold
        4. Identify original features exceeding the threshold
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            model_params: Optional model parameters (uses config defaults if None)
            
        Returns:
            Tuple of (threshold, significant_features, shadow_importances)
        """
        try:
            if not self.config.enable_gini_significance:
                logger.info("Gini significance testing disabled in configuration")
                return 0.0, [], {}
            
            logger.info("Starting Gini importance significance testing with shadow features")
            
            # Create shadow features
            X_shadow = self.create_shadow_features(X_train)
            
            # Combine original and shadow features
            X_augmented = pd.concat([X_train, X_shadow], axis=1)
            logger.info(f"Created augmented dataset with {len(X_augmented.columns)} features")
            
            # Initialize model with same parameters as main analysis
            if self.config.model_type == 'classifier':
                model = RandomForestClassifier(
                    n_estimators=200, 
                    class_weight='balanced', 
                    random_state=self.random_state, 
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200, 
                    random_state=self.random_state, 
                    n_jobs=-1
                )
            
            # Train model on augmented dataset
            logger.info("Training Random Forest on augmented dataset...")
            model.fit(X_augmented, y_train)
            
            # Extract importance scores
            all_importances = dict(zip(X_augmented.columns, model.feature_importances_))
            
            # Separate original and shadow importances
            original_features = X_train.columns.tolist()
            shadow_features = X_shadow.columns.tolist()
            
            shadow_importances = {feat: all_importances[feat] for feat in shadow_features}
            original_importances = {feat: all_importances[feat] for feat in original_features}
            
            # Calculate significance threshold
            threshold = self.calculate_significance_threshold(shadow_importances)
            
            # Identify significant features
            significant_features = [
                feat for feat, importance in original_importances.items() 
                if importance > threshold
            ]
            
            logger.info(f"Found {len(significant_features)} significant features out of {len(original_features)}")
            logger.info(f"Significant features: {significant_features}")
            
            return threshold, significant_features, shadow_importances
            
        except Exception as e:
            logger.error(f"Gini significance testing failed: {str(e)}")
            # Return safe defaults to allow analysis to continue
            return 0.0, [], {}
    
    def test_shap_significance(self, shap_values: np.ndarray, feature_names: List[str]) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
        """
        Test SHAP value significance using Wilcoxon signed-rank tests with FDR correction.
        
        This method implements rigorous statistical testing for SHAP values:
        1. Perform one-sample Wilcoxon signed-rank test for each feature
        2. Test null hypothesis: median SHAP value = 0
        3. Apply Benjamini-Hochberg FDR correction for multiple comparisons
        4. Identify features with adjusted p-values < alpha
        
        Args:
            shap_values: SHAP values matrix (n_samples x n_features)
            feature_names: List of feature names corresponding to SHAP values
            
        Returns:
            Tuple of (raw_pvalues, adjusted_pvalues, significant_features)
        """
        try:
            if not self.config.enable_shap_significance:
                logger.info("SHAP significance testing disabled in configuration")
                return {}, {}, []
            
            logger.info(f"Starting SHAP significance testing for {len(feature_names)} features")
            
            raw_pvalues = {}
            
            # Perform Wilcoxon signed-rank test for each feature
            for i, feature_name in enumerate(feature_names):
                feature_shap_values = shap_values[:, i]
                
                # Skip features with all zero SHAP values
                if np.all(feature_shap_values == 0):
                    logger.warning(f"Feature '{feature_name}' has all zero SHAP values, skipping test")
                    raw_pvalues[feature_name] = 1.0
                    continue
                
                try:
                    # One-sample Wilcoxon signed-rank test (H0: median = 0)
                    statistic, p_value = wilcoxon(feature_shap_values, alternative='two-sided')
                    raw_pvalues[feature_name] = p_value
                    
                except ValueError as e:
                    # Handle cases where test cannot be performed (e.g., all values identical)
                    logger.warning(f"Wilcoxon test failed for feature '{feature_name}': {str(e)}")
                    raw_pvalues[feature_name] = 1.0
            
            logger.info(f"Completed Wilcoxon tests for {len(raw_pvalues)} features")
            
            # Apply FDR correction using existing utilities
            p_values_list = list(raw_pvalues.values())
            adjusted_p_values_list = apply_fdr_correction(p_values_list, alpha=self.config.significance_alpha)
            
            # Create adjusted p-values dictionary
            adjusted_pvalues = dict(zip(feature_names, adjusted_p_values_list))
            
            # Identify significant features
            significant_features = [
                feature for feature, adj_p in adjusted_pvalues.items()
                if adj_p < self.config.significance_alpha
            ]
            
            logger.info(f"Found {len(significant_features)} SHAP-significant features after FDR correction")
            logger.info(f"SHAP-significant features: {significant_features}")
            
            return raw_pvalues, adjusted_pvalues, significant_features
            
        except Exception as e:
            logger.error(f"SHAP significance testing failed: {str(e)}")
            # Return safe defaults to allow analysis to continue
            return {}, {}, []
    
    def run_all_tests(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     shap_values: np.ndarray, feature_names: List[str]) -> SignificanceResults:
        """
        Run both Gini and SHAP significance tests and return comprehensive results.
        
        This is the main entry point for significance testing that orchestrates
        both shadow feature testing for Gini importance and Wilcoxon testing for SHAP values.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            shap_values: SHAP values matrix
            feature_names: List of feature names
            
        Returns:
            SignificanceResults object containing all test results
        """
        try:
            logger.info("Starting comprehensive feature importance significance testing")
            
            # Test Gini significance
            gini_threshold, gini_significant, shadow_importances = self.test_gini_significance(
                X_train, y_train
            )
            
            # Test SHAP significance
            shap_pvalues, shap_adjusted_pvalues, shap_significant = self.test_shap_significance(
                shap_values, feature_names
            )
            
            # Create results object
            results = SignificanceResults(
                gini_threshold=gini_threshold,
                gini_significant_features=gini_significant,
                shadow_importances=shadow_importances,
                shap_pvalues=shap_pvalues,
                shap_adjusted_pvalues=shap_adjusted_pvalues,
                shap_significant_features=shap_significant,
                alpha_level=self.config.significance_alpha,
                n_features_tested=len(feature_names),
                n_shadow_features=len(shadow_importances)
            )
            
            self.results = results
            logger.info("Significance testing completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive significance testing failed: {str(e)}")
            # Return minimal results to allow analysis to continue
            return SignificanceResults(
                gini_threshold=0.0,
                gini_significant_features=[],
                shadow_importances={},
                shap_pvalues={},
                shap_adjusted_pvalues={},
                shap_significant_features=[],
                alpha_level=self.config.significance_alpha,
                n_features_tested=len(feature_names),
                n_shadow_features=0
            )