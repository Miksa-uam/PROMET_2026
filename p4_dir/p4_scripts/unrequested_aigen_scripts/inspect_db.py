import sqlite3

def inspect_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", [t[0] for t in tables])
    print("-" * 50)
    
    for table_tuple in tables:
        table_name = table_tuple[0]
        print(f"\nSchema for table: {table_name}")
        
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
            
        print(f"\nSample of 3 rows from {table_name}:")
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            rows = cursor.fetchall()
            for row in rows:
                print(f"  {row}")
        except Exception as e:
            print(f"Error reading sample: {e}")
        
        print("-" * 50)
        
    conn.close()

if __name__ == "__main__":
    db_path = "dbs/pnk_db2_filtered.sqlite"
    inspect_db(db_path)
