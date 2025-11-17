import sqlite3
import pandas as pd

# Check cluster database structure
conn = sqlite3.connect('dbs/pnk_db2_p2_cluster_pam_goldstd.sqlite')

# Get tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables in cluster database:")
print(tables)
print()

# Check first table
if len(tables) > 0:
    table_name = tables.iloc[0]['name']
    print(f"Sample from '{table_name}':")
    df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(df)
    print()
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {len(pd.read_sql_query(f'SELECT * FROM {table_name}', conn))}")

conn.close()
