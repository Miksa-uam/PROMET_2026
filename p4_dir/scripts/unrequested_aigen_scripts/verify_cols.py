import sqlite3

def check_columns():
    conn = sqlite3.connect("dbs/pnk_db2_filtered.sqlite")
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(measurements_filtered);")
        cols = cursor.fetchall()
        print("Columns in measurements_filtered:")
        for c in cols:
            print(c)
    except Exception as e:
        print("Error:", e)
    conn.close()

if __name__ == "__main__":
    check_columns()
