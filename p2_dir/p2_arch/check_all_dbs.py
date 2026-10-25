import sqlite3
import pandas as pd

print("="*60)
print("CHECKING DATABASE STRUCTURES")
print("="*60)

# Check main input database
print("\n1. pnk_db2_p2_in.sqlite:")
conn = sqlite3.connect('dbs/pnk_db2_p2_in.sqlite')
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(f"   Tables: {tables['name'].tolist()}")
conn.close()

# Check cluster database
print("\n2. pnk_db2_p2_cluster_pam_goldstd.sqlite:")
conn = sqlite3.connect('dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite')
df = pd.read_sql_query("SELECT * FROM clust_labels_jaccard_wgc_pam_goldstd LIMIT 3", conn)
print(f"   Cluster label columns: {[c for c in df.columns if 'pam_k' in c]}")
print(f"   Sample cluster assignments (k=7): {df['pam_k7'].tolist()}")
conn.close()

# Check if there's a merged table
print("\n3. Looking for merged cluster + outcome data...")
conn = sqlite3.connect('dbs/pnk_db2_p2_in.sqlite')
try:
    # Try to find a table with cluster_id
    for table in tables['name']:
        df_check = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 1", conn)
        if 'cluster_id' in df_check.columns or any('cluster' in col.lower() for col in df_check.columns):
            print(f"   Found cluster column in: {table}")
            print(f"   Columns: {df_check.columns.tolist()}")
except:
    print("   No merged table found in p2_in.sqlite")
conn.close()

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("Cluster data structure:")
print("  - Cluster assignments: separate database (pnk_db2_p2_cluster_*.sqlite)")
print("  - Outcome data: pnk_db2_p2_in.sqlite (timetoevent tables)")
print("  - Need to JOIN: medical_record_id from both databases")
print("="*60)
