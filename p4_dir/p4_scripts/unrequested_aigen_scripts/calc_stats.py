import sqlite3
import pandas as pd
import numpy as np

def calculate_stats():
    conn = sqlite3.connect('dbs/pnk_db2_filtered.sqlite')
    query = "SELECT patient_id, COUNT(*) as count FROM measurements_filtered GROUP BY patient_id"
    df = pd.read_sql_query(query, conn)
    conn.close()

    avg = df['count'].mean()
    sd = df['count'].std()

    # Top decile (top 10%)
    threshold = df['count'].quantile(0.90)
    top_decile_df = df[df['count'] >= threshold]
    top_avg = top_decile_df['count'].mean()
    top_sd = top_decile_df['count'].std()

    print(f"Total Patients: {len(df)}")
    print(f"Overall Average: {avg:.2f}")
    print(f"Overall SD: {sd:.2f}")
    print(f"Top Decile Threshold (>=): {threshold}")
    print(f"Top Decile Average: {top_avg:.2f}")
    print(f"Top Decile SD: {top_sd:.2f}")
    print(f"Maximum Entries for a single patient: {df['count'].max()}")

if __name__ == "__main__":
    calculate_stats()
