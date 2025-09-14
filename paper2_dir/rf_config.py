# In a new file: rf_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class RFAnalysisConfig:
    """Configuration for a complete Random Forest feature importance analysis."""
    
    # --- Core Parameters ---
    analysis_name: str
    outcome_variable: str
    model_type: str # 'classifier' or 'regressor'
    predictors: List[str]
    covariates: List[str] = field(default_factory=list)

    # --- Classifier-Specific Parameters ---
    classifier_threshold: Optional[float] = None
    threshold_direction: Optional[str] = None # 'greater_than_or_equal' or 'less_than_or_equal'

    # --- Data & Output Paths ---
    db_path: str = "C:/Users/Felhasználó/Desktop/Projects/PNK_DB2/paper2_dir/pnk_db2_p2_in.sqlite"
    input_table: str = "timetoevent_wgc_compl"
    output_dir: str = "C:/Users/Felhasználó/Desktop/Projects/PNK_DB2/paper2_dir/rf_outputs"

    # --- Plotting & Labels ---
    nice_names: Dict[str, str] = field(default_factory=lambda: {
        "age": "Age (years)",
        "sex_f": "Sex (Female)",
        "baseline_bmi": "Baseline BMI",
        "womens_health_and_pregnancy": "Women's health/pregnancy",
        "mental_health": "Mental health",
        "family_issues": "Family issues",
        "medication_disease_injury": "Medication/disease/injury",
        "physical_inactivity": "Physical inactivity",
        "eating_habits": "Eating habits",
        "schedule": "Schedule",
        "smoking_cessation": "Smoking cessation",
        "treatment_discontinuation_or_relapse": "Treatment relapse",
        "pandemic": "COVID-19 pandemic",
        "lifestyle_circumstances": "Lifestyle circumstances",
        "none_of_above": "None of the above"
    })

    def __post_init__(self):
        """Validate configuration after creation."""
        if self.model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'")
        if self.model_type == 'classifier' and (self.classifier_threshold is None or self.threshold_direction is None):
            raise ValueError("Classifier requires a threshold and direction.")
