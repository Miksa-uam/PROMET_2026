import sys
sys.path.append('.')
from wgc_mother_cohort_lollipop import main

# Test with a few key variables to check p-value handling
test_variables = [
    'Total weight loss (%)',      # Should be ** (FDR significant)
    'Instant dropouts (n)',       # Should be * (raw significant only)
    '40-day dropouts (n)',        # Should be ** (FDR significant)
    'Age (years)',                # Should be ** (FDR significant)
    'Follow-up length (days)'     # Should be no significance
]

print("Testing p-value fix...")
pct_data, summary_table = main(
    db_path="../dbs/pnk_db2_p2_out.sqlite",
    table_name="wgc_cmpl_dmgrph_strt",
    variables_to_include=test_variables,
    output_plot_path="../outputs/test_pvalue_fix.png",
    output_table_path="../outputs/test_pvalue_fix.csv"
)

print("\nP-value results:")
for _, row in pct_data.iterrows():
    print(f"{row['variable']}: {row['percent_change']:+.1f}% | Raw p={row['p_value_raw']:.4f} | FDR p={row['p_value_fdr']:.4f} | Sig='{row['significance']}'")