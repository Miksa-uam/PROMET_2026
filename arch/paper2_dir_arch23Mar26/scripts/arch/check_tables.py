import sqlite3
import os

# Check the correct database locations based on descriptive_comparisons.py
db_paths = [
    '../dbs/pnk_db2_p2_in.sqlite',   # Input database (raw cohort data)
    '../dbs/pnk_db2_p2_out.sqlite'   # Output database (comparison results)
]

for db_path in db_paths:
    if os.path.exists(db_path):
        print(f'\n=== Database: {db_path} ===')
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if tables:
                print('Available tables:')
                for table in tables:
                    print(f'  {table[0]}')
            else:
                print('No tables found')
            
            conn.close()
        except Exception as e:
            print(f'Error accessing database: {e}')
    else:
        print(f'\n=== Database: {db_path} === (NOT FOUND)')