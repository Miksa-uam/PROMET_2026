# p2_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

"""
1. DATA PREPARATION CONFIGS
"""

@dataclass
class paths_config:
    source_dir: str # Raw and pre-processed PNK databases - DB2_standard folder
    source_db: str # The specific, filtered source database - pnk_db2_filtered
    paper_dir: str # Paper-specific files and code folder
    # paper_filter_criteria: str
    paper_in_db: str  # The input database created and used in a specific research paper
    paper_out_db: str  # The output database created and used in a specific research paper

@dataclass
class filtering_config:
    filtering_sql_query: str # SQL query to select patient-medical record combinations for paper-specific analysis

@dataclass
class timetoevent_config:
    input_measurements: str # The input measurements used in the time-to-event type table
    input_records: str # The input medical records used in the time-to-event type table
    output_table: str # The output table for the time-to-event type analysis, within the output database
    weight_loss_targets: List[int] # A configurable list of % weight loss targets - eg. 5/10% WL
    time_windows: List[int] # A configurable list of follow-up time windows - eg. 40/60 days follow-up
    window_span: int # The span of the follow-up time windows - eg. 40 +/- 10 days
    fetch_from_records: List[str] # Columns to fetch from the medical records table
    followup_columns: List[str] # Columns containing follow-up data
    predictor_columns: List[str] # Columns containing predictor data

@dataclass
class timetoevent_subsetting_config:
    source_table: str
    definitions: Dict[str, List[str]] = field(default_factory=dict)

"""
2. DATA ANALYSIS CONFIGS
"""

@dataclass
class descriptive_comparisons_config:
    """A streamlined config for a single descriptive comparison analysis."""
    analysis_name: str
    input_cohort_name: str
    mother_cohort_name: str
    
    # Row order for tables (variable name and pretty name)
    row_order: List[Tuple[str, str]]
    
    # Settings for demographic stratification
    demographic_output_table: str
    demographic_strata: List[str]
    
    # Settings for WGC stratification
    wgc_output_table: str
    wgc_strata: List[str] # This can be left empty if not needed

    # Optional plot filename
    bias_plot_filename: Optional[str] = None

@dataclass
class paper2_rf_config:
    """Configuration for a complete Random Forest feature importance analysis for the second paper."""
    
    # --- Core Parameters ---
    analysis_name: str
    outcome_variable: str
    model_type: str # 'classifier' or 'regressor'
    predictors: List[str]
    covariates: List[str] = field(default_factory=list)

    # --- Model Hyperparameter Tuning ---
    run_hyperparameter_tuning: bool = False # Default is OFF
    
    # --- Classifier-Specific Parameters ---
    classifier_threshold: Optional[float] = None
    threshold_direction: Optional[str] = None # 'greater_than_or_equal' or 'less_than_or_equal'

    # --- Data & Output Paths ---
    db_path: str = "../dbs/pnk_db2_p2_in.sqlite"
    input_table: str = "timetoevent_wgc_compl"
    output_dir: str = "../outputs/rf_outputs"

    # --- Plotting & Labels ---
    nice_names: Dict[str, str] = field(default_factory=lambda: {
        "age": "Age (years)",
        "sex_f": "Sex",
        "baseline_bmi": "Baseline BMI",
        "womens_health_and_pregnancy": "Women's health or pregnancy",
        "mental_health": "Mental health",
        "family_issues": "Family issues",
        "medication_disease_injury": "Medication, disease or injury",
        "physical_inactivity": "Physical inactivity",
        "eating_habits": "Eating habits",
        "schedule": "Schedule",
        "smoking_cessation": "Smoking cessation",
        "treatment_discontinuation_or_relapse": "Treatment discontinuation or relapse",
        "pandemic": "COVID-19 pandemic",
        "lifestyle_circumstances": "External circumstances",
        "none_of_above": "None of the above"
    })

    def __post_init__(self):
        """Validate configuration after creation."""
        if self.model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'")
        if self.model_type == 'classifier' and (self.classifier_threshold is None or self.threshold_direction is None):
            raise ValueError("Classifier requires a threshold and direction.")

"""
0. MASTER CONFIG
"""

@dataclass
class master_config:
    """
    Main container of config objects. 
    Some are analysis-specific, so we leave these with a default value of None, 
    that only needs to be set when the specific analysis is run.
    """
    paths: Optional[paths_config] = None
    filtering: Optional[filtering_config] = None
    timetoevent: Optional[timetoevent_config] = None
    timetoevent_subsetting: Optional[timetoevent_subsetting_config] = None
    descriptive_comparisons: Optional[descriptive_comparisons_config] = None
    paper2_rf: Optional[paper2_rf_config] = None

