# p2_config.py
from dataclasses import dataclass

@dataclass
class paper2_paths:
    source_directory: str # Raw and pre-processed PNK databases - DB2_standard folder
    source_db_path: str # The specific, filtered source database - pnk_db2_filtered
    paper2_directory: str # Paper 2-specific files and code
    p2_in_db_path: str  # The input database created and used in Paper 2 - pnk_db2_p2_in
    p2_out_db_path: str  # The output database created and used in Paper 2 - pnk_db2_p2_out