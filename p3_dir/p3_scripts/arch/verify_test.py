
import sqlite3
import pandas as pd

db_path = '../dbs/pnk_db2_genomics_in_test.sqlite'
conn = sqlite3.connect(db_path)
df = pd.read_sql("SELECT genomics_results_date, baseline_date, final_date, genomics_within_record, genomics_3wk_within_record_start, genomics_before_record, genomics_after_record FROM timetoevent_all WHERE genomics_results_date IS NOT NULL LIMIT 20", conn)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))
conn.close()
