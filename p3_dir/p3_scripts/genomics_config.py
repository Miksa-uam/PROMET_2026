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

@dataclass
class OmnibusTableConfig:
    """Multi-group Kruskal-Wallis / Chi² descriptive table.

    Parameters
    ----------
    paths           : paths_config
    cohort_tables   : {sql_table_name: group_code}
    row_order       : [(col_name, pretty_label), ...]  — defines rows AND columns to load
    output_table    : name of the SQLite table written to paper_out_db
    display         : {group_code: display_label}  — used as column headers in saved table
    fdr_correction  : apply Benjamini-Hochberg FDR correction (default True)
    """
    paths:          object
    cohort_tables:  Dict[str, str]
    row_order:      List[Tuple[str, str]]
    output_table:   str
    display:        Dict[str, str]
    fdr_correction: bool = True


@dataclass
class OmnibusVizConfig:
    """Multi-group visualisation suite: alluvial, KM, and violin plots.

    Column names for adherence and outcome variables are derived automatically
    from landmark_day and wl_target — you never hard-code column names here.

    Parameters
    ----------
    paths                   : paths_config
    cohort_tables           : {sql_table_name: group_code}
    group_colors            : {group_code: hex_color}
    master_group_order      : ordered list of all possible group_codes;
                              only those present in data are rendered
    display                 : {group_code: display_label}
                              Also used for auto-generated adherence / WL labels
                              if you want to override the defaults.
    output_dir              : folder where HTML files are written (created if absent)
    outputs                 : {'alluvial': filename, 'km': filename, 'violin': filename}
                              Omit a key entirely to skip that plot.

    Landmark / outcome settings
    ---------------------------
    landmark_day            : N — used to look up Nd_dropout and Nd_wl_% columns
    wl_target               : T — used to look up T%_wl_achieved and days_to_T%_wl
    include_instant_dropout : if True, loads and displays instant_dropout as a
                              separate tier on the alluvial adherence axis

    KM
    --
    km_time_col             : follow-up time column (default 'total_followup_days')

    Plot titles / subtitles
    -----------------------
    All optional. violin_title auto-fills from landmark_day if left blank.
    """
    paths:                  object
    cohort_tables:          Dict[str, str]
    group_colors:           Dict[str, str]
    master_group_order:     List[str]
    display:                Dict[str, str]
    output_dir:             str
    outputs:                Dict[str, str]

    landmark_day:            int  = 120
    wl_target:               int  = 10
    include_instant_dropout: bool = True
    km_time_col:             str  = "total_followup_days"

    alluvial_title:    str = "Patient Flow: Personalization Timing, Adherence and Weight Loss"
    km_title:          str = "Kaplan-Meier Survival Curves by Personalization Group"
    violin_title:      str = ""   # auto-filled from landmark_day when blank
    alluvial_subtitle: str = "\nFirst medical records only"
    km_subtitle:       str = "\nFirst medical records only | Shaded area = 95% CI"
    violin_subtitle:   str = "\nViolin width ∝ % reaching landmark | Box = IQR"

    # ── Derived column names — read-only, computed from landmark_day / wl_target ──
    @property
    def dropout_col(self) -> str:
        return f"{self.landmark_day}d_dropout"

    @property
    def wl_pct_col(self) -> str:
        return f"{self.landmark_day}d_wl_%"

    @property
    def wl_achieved_col(self) -> str:
        return f"{self.wl_target}%_wl_achieved"

    @property
    def days_to_wl_col(self) -> str:
        return f"days_to_{self.wl_target}%_wl"

    @property
    def cols_alluvial(self) -> List[str]:
        cols = [self.dropout_col, self.wl_achieved_col]
        if self.include_instant_dropout:
            cols.insert(0, "instant_dropout")
        return cols

    @property
    def cols_km(self) -> List[str]:
        return [self.km_time_col, self.wl_achieved_col, self.days_to_wl_col]

    @property
    def cols_violin(self) -> List[str]:
        return [self.dropout_col, self.wl_pct_col]


"""
0. MASTER CONFIG
"""

@dataclass
class master_config:
    """
    Main container of config objects. 
    Some are analysis-specific, so we leave these with a default value of None, 
    that only needs to be set when the specific analysis is run.
    Omnibus table and viz configs are instantiated independently in their own
    notebook cells — they do not belong here.
    """
    paths: Optional[paths_config] = None
    filtering: Optional[filtering_config] = None
    timetoevent: Optional[timetoevent_config] = None
    timetoevent_subsetting: Optional[timetoevent_subsetting_config] = None
    descriptive_comparisons: Optional[List[descriptive_comparisons_config]] = None
