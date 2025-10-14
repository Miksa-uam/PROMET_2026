import sqlite3
import pandas as pd

conn = sqlite3.connect('../dbs/pnk_db2_p2_out.sqlite')

# Get column names first
df_cols = pd.read_sql_query("PRAGMA table_info(wgc_cmpl_dmgrph_strt)", conn)
print("Available columns:")
for col in df_cols['name']:
    print(f"  {col}")

print("\n" + "="*50)

# Check specific row for Instant dropouts
df = pd.read_sql_query("""
    SELECT Variable, 
           [Cohort comparison: p-value] as raw_p,
           [Cohort comparison: p-value (FDR-corrected)] as fdr_p
    FROM wgc_cmpl_dmgrph_strt 
    WHERE Variable = 'Instant dropouts (n)'
""", conn)

print("Instant dropouts p-values:")
print(df)

print("\n" + "="*50)

# Check a few more examples
df_sample = pd.read_sql_query("""
    SELECT Variable, 
           [Cohort comparison: p-value] as raw_p,
           [Cohort comparison: p-value (FDR-corrected)] as fdr_p
    FROM wgc_cmpl_dmgrph_strt 
    WHERE Variable IN ('Instant dropouts (n)', '40-day dropouts (n)', 'Total weight loss (%)')
""", conn)

print("Sample p-values:")
print(df_sample)

conn.close()