import sqlite3
import os

db_path = r"C:\Users\Felhasználó\Desktop\Projects\PNK_DB2\genomics_dir\dbs\pnk_db2_filtered.sqlite"

try:
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Total unique patient_ids
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM alleles_filtered")
    total_unique = cursor.fetchone()[0]

    # 2. Unique patient_ids where allele IS NOT NULL
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM alleles_filtered WHERE allele IS NOT NULL")
    unique_not_null_allele = cursor.fetchone()[0]

    # 3. Unique patient_ids where rs DOES NOT START with 'rs'
    # using NOT LIKE 'rs%'
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM alleles_filtered WHERE rs NOT LIKE 'rs%'")
    unique_no_rs_prefix = cursor.fetchone()[0]
    
    # Also verifying if there are any that DO start with 'rs' just for sanity (optional but good context)
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM alleles_filtered WHERE rs LIKE 'rs%'")
    unique_rs_prefix = cursor.fetchone()[0]

    print(f"RES_TOTAL_UNIQUE:{total_unique}")
    print(f"RES_ALLELE_NOT_NULL:{unique_not_null_allele}")
    print(f"RES_RS_NO_PREFIX:{unique_no_rs_prefix}")

    conn.close()

except Exception as e:
    print(f"An error occurred: {e}")
