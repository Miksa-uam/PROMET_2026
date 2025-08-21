# p2_config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class paths_config:
    source_dir: str # Raw and pre-processed PNK databases - DB2_standard folder
    source_db: str # The specific, filtered source database - pnk_db2_filtered
    paper_dir: str # Paper-specific files and code folder
    paper_in_db: str  # The input database created and used in a specific research paper
    paper_out_db: str  # The output database created and used in a specific research paper
    
@dataclass
class timetoevent_config:
    # paper_in_db: str  # The input database created and used in a specific research paper
    # paper_out_db: str  # The output database created and used in a specific research paper
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
class master_config:
    """
    Main container of config objects. 
    Some are analysis-specific, so we leave these with a default value of None, 
    that only needs to be set when the specific analysis is run.
    """
    paths: paths_config = None
    timetoevent: timetoevent_config = None
