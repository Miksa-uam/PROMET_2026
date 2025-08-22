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
    # this should be optional
    descriptive_comparisons: List[descriptive_comparisons_config] = field(default_factory=list)

