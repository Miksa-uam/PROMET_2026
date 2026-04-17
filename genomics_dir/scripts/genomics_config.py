# genomics_config.py
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
    paper_in_db: str  # The input database created and used in the genomics project
    paper_out_db: str  # The output database created and used in the genomics project

@dataclass
class filtering_config:
    filtering_sql_query: str # SQL query to select patient-medical record combinations for paper-specific analysis

@dataclass
class timetoevent_config:
    input_measurements: str # The input measurements used in the time-to-event type table
    input_records: str # The input medical records used in the time-to-event type table
    input_alleles: str # The input alleles table used in the time-to-event type table
    output_table: str # The output table for the time-to-event type analysis, within the output database
    weight_loss_targets: List[int] # A configurable list of % weight loss targets - eg. 5/10% WL
    time_windows: List[int] # A configurable list of follow-up time windows - eg. 40/60 days follow-up
    window_span: int # The span of the follow-up time windows - eg. 40 +/- 10 days
    fetch_from_records: List[str] # Columns to fetch from the medical records table
    fetch_from_alleles: List[str] # Columns to fetch from the alleles table
    clinical_data_columns: List[str] # Columns containing clinical data
    metadata_columns: List[str] # Columns containing metadata

@dataclass
class timetoevent_subsetting_config:
    source_table: str
    definitions: Dict[str, List] = field(default_factory=dict)

"""
2. DATA ANALYSIS CONFIGS
"""

@dataclass
class descriptive_comparisons_config:
    """A streamlined config for a single descriptive comparison analysis.
    
    Parameters:
        analysis_name: Name identifier for the analysis
        input_cohort_name: Name of the input cohort being analyzed
        mother_cohort_name: Name of the reference/mother cohort for comparison
        row_order: List of tuples containing (variable_name, pretty_name) for table row ordering
        demographic_output_table: Output table name for demographic stratification results
        demographic_strata: List of demographic variables to stratify by
        wgc_output_table: Output table name for weight gain cause stratification results
        wgc_strata: List of weight gain causes to stratify by (can be empty if not needed)
        bias_plot_filename: Optional filename for bias plot output
        wgc_vs_mean_output_table: Optional output table name for WGC vs population mean analysis
        cluster_vs_mean_output_table: Optional output table name for cluster vs population mean analysis
        fdr_correction: Whether to apply False Discovery Rate correction using Benjamini-Hochberg method.
                       When True, adds FDR-corrected p-value columns with "(FDR-corrected)" suffix.
                       When False (default), behaves as original implementation with raw p-values only.
                       Maintains backward compatibility with existing analysis configurations.
    """
    analysis_name: str
    input_cohort_name: str
    mother_cohort_name: str
    
    # Row order for tables (variable name and pretty name)
    row_order: List[Tuple[str, str]]
    
    # Settings for demographic stratification
    demographic_output_table: str
    demographic_strata: List[str]
    
    # Optional plot filename
    # bias_plot_filename: Optional[str] = None
    
    # FDR correction setting
    fdr_correction: bool = False

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
    descriptive_comparisons: Optional[List[descriptive_comparisons_config]] = None
