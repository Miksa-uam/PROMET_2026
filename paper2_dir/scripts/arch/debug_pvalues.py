import sys
sys.path.append('.')
from wgc_mother_cohort_lollipop import extract_comparison_data, calculate_percentage_changes

# Extract data and check what we get
db_path = "../dbs/pnk_db2_p2_out.sqlite"
table_name = "wgc_cmpl_dmgrph_strt"

df = extract_comparison_data(db_path, table_name)

# Check specific row
instant_row = df[df['Variable'] == 'Instant dropouts (n)']
if not instant_row.empty:
    row = instant_row.iloc[0]
    print("Instant dropouts row data:")
    print(f"Variable: {row['Variable']}")
    print(f"Raw p-value: {row.get('Cohort comparison: p-value', 'NOT FOUND')}")
    print(f"FDR p-value: {row.get('Cohort comparison: p-value (FDR-corrected)', 'NOT FOUND')}")
    print(f"Raw p-value type: {type(row.get('Cohort comparison: p-value'))}")
    print(f"FDR p-value type: {type(row.get('Cohort comparison: p-value (FDR-corrected)'))}")

print("\n" + "="*50)

# Now check what calculate_percentage_changes produces
pct_data = calculate_percentage_changes(df)
instant_pct = pct_data[pct_data['variable'] == 'Instant dropouts (n)']
if not instant_pct.empty:
    row = instant_pct.iloc[0]
    print("After calculate_percentage_changes:")
    print(f"Variable: {row['variable']}")
    print(f"Raw p-value: {row['p_value_raw']}")
    print(f"FDR p-value: {row['p_value_fdr']}")
    print(f"Significance: '{row['significance']}'")
else:
    print("Instant dropouts not found in percentage changes data!")