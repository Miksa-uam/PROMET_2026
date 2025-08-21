# p2_dataprep.py

import sqlite3
import pandas as pd
from p2_config import paper2_paths

def create_paper2_subset(paths: paper2_paths) -> None:
    """Create a Paper 2-specific SQL database subset"""
    # Connect to the source database
    source_conn = sqlite3.connect(paths.source_db_path)

    # Identify patient-medical record combinations to include in the analysis. 
    # To identify patients that satisfy the following criteria:
    # gdpr4 = 1, gdpr10 = 0 and medical_record_sequence = 1 in medical_records_filtered -> to only use first records of PROMET CONNECT patients. 
    # query: 
        # WHERE gdpr4 = 1
        #   AND gdpr10 = 0
        #   AND medical_record_sequence = 1
    sql_select_ids = """
        SELECT medical_record_id, patient_id
        FROM medical_records_filtered
    """

    records = pd.read_sql_query(sql_select_ids, source_conn)
    record_ids = tuple(records['medical_record_id'])
    patient_ids = tuple(records['patient_id'])

    # Pull rows from both tables corresponding to the identified records.
    # Create new table names measurements_p2 and medical_records_p2.
    table_mapping = {
        "medical_records_filtered": ("medical_records_p2", record_ids),
        "measurements_filtered": ("measurements_p2", record_ids)
    }

    # Create new SQLite database connection to write filtered data.
    p2_in_conn = sqlite3.connect(paths.p2_in_db_path)

    for src_table, (dst_table, mr_ids) in table_mapping.items():
        # Prepare filtering query for tables that include 'medical_record_id'
        if src_table in ("medical_records_filtered", "measurements_filtered"):
            query = f"""
                SELECT *
                FROM {src_table}
                WHERE medical_record_id IN {mr_ids}
            """
        else:
            # For other tables (if needed), one might filter by patient_id.
            query_check_column = f"PRAGMA table_info({src_table});"
            columns = pd.read_sql_query(query_check_column, source_conn)
            if 'patient_id' not in columns['name'].values:
                continue  # Skip if no patient_id column.
            query = f"""
                SELECT *
                FROM {src_table}
                WHERE patient_id IN {patient_ids}
            """
        # Execute the query and write into the output database.
        df_filtered = pd.read_sql_query(query, source_conn)
        df_filtered.to_sql(dst_table, p2_in_conn, index=False, if_exists="replace")
        print(f"Wrote {len(df_filtered)} rows from {src_table} to {dst_table}.")

    # Close both connections
    p2_in_conn.close()
    source_conn.close()

    # Print summary
    # print(f"Filtered data has been saved to {new_db_path}.")
    # print(f"Total first-record combinations: {len(records)} from {len(set(records['patient_id']))} patients.")

    print(f"Data has been saved to {paths.p2_in_db_path}.")
    print(f"Total records: {len(records)} from {len(set(records['patient_id']))} patients.")