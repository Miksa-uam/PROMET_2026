
import sqlite3
import pandas as pd
import numpy as np
import os

# Adjust path as needed. Assuming notebook is in 'scripts/' and DB in 'dbs/'
db_path = '../dbs/pnk_db2_genomics_in_test.sqlite'

def add_genomics_vars(table_name):
    print(f"Processing {table_name}...")
    conn = sqlite3.connect(db_path)
    
    try:
        # Read table
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        print(f"Read {len(df)} rows.")
        
        # Ensure date columns are datetime
        date_cols = ['genomics_results_date', 'baseline_date', 'final_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # 1. genomics_within_record
        # 1 if baseline <= result <= final, else 0
        df['genomics_within_record'] = 0
        mask_within = (df['genomics_results_date'] >= df['baseline_date']) & (df['genomics_results_date'] <= df['final_date'])
        df.loc[mask_within, 'genomics_within_record'] = 1
        
        # 2. genomics_3wk_within_record_start
        # If genomics_within_record == 0 (or not 1), NULL.
        # If within==1:
        #    if result <= baseline + 21 days -> 1
        #    else (result > baseline + 21 days) -> 0
        df['genomics_3wk_within_record_start'] = np.nan
        
        # Calculate 21 day mark
        cutoff = df['baseline_date'] + pd.Timedelta(days=21)
        
        # Masks
        mask_w1 = df['genomics_within_record'] == 1
        mask_early = mask_w1 & (df['genomics_results_date'] <= cutoff)
        mask_late = mask_w1 & (df['genomics_results_date'] > cutoff)
        
        df.loc[mask_early, 'genomics_3wk_within_record_start'] = 1
        df.loc[mask_late, 'genomics_3wk_within_record_start'] = 0
        
        # 3. genomics_before_record
        # 1 if result < baseline, else 0
        df['genomics_before_record'] = 0
        mask_before = df['genomics_results_date'] < df['baseline_date']
        df.loc[mask_before, 'genomics_before_record'] = 1
        
        # 4. genomics_after_record
        # 1 if result > final, else 0
        df['genomics_after_record'] = 0
        mask_after = df['genomics_results_date'] > df['final_date']
        df.loc[mask_after, 'genomics_after_record'] = 1
        
        # Convert generated columns to numeric/nullable types if needed
        # Int64 allows for integers with NaN? No, float is standard for NaN in pandas < 1.0 but nullable int exists now.
        # To match "indicated with 1/0 and sometimes null", float with 1.0/0.0/NaN is safest for SQLite compatibility via pandas.
        # But 'genomics_within_record' etc are 0/1 always, so can be int.
        
        # Reorder columns
        new_cols = ['genomics_within_record', 'genomics_3wk_within_record_start', 'genomics_before_record', 'genomics_after_record']
        cols = list(df.columns)
        for c in new_cols:
            if c in cols: cols.remove(c) # Should be there at end
            
        # Target index: after genomics_results_date
        target_col = 'genomics_results_date'
        try:
            target_idx = cols.index(target_col) + 1
        except ValueError:
            target_idx = len(cols)
            print(f"Warning: {target_col} not found, appending to end.")
            
        for c in reversed(new_cols):
            cols.insert(target_idx, c)
            
        df = df[cols]
        
        # Write back
        # Note: dates might be written as 'YYYY-MM-DD HH:MM:SS' strings automatically or timestamps.
        # Verify date format.
        # print("Sample row:\n", df.iloc[0])
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Successfully updated {table_name}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_genomics_vars('timetoevent_all')
