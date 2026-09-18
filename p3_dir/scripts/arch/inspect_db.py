
import sqlite3
import pandas as pd
import os

db_path = r'..\dbs\pnk_db2_genomics_in.sqlite'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Tables:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(tables)
    
    for t in tables:
        table_name = t[0]
        if 'time' in table_name.lower() or 'event' in table_name.lower():
            print(f"\nSchema for {table_name}:")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for c in columns:
                print(c)
                
    conn.close()
except Exception as e:
    print(e)
