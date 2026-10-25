import sqlite3
import pandas as pd

conn = sqlite3.connect('../dbs/pnk_db2_p2_out.sqlite')
df = pd.read_sql_query("""
    SELECT Variable 
    FROM wgc_cmpl_dmgrph_strt 
    WHERE Variable NOT LIKE 'delim_%' 
    AND Variable != 'N' 
    AND Variable IS NOT NULL
""", conn)

print('Available variables in comparison table:')
for i, var in enumerate(df['Variable'].tolist(), 1):
    print(f'{i:2d}. {var}')

conn.close()