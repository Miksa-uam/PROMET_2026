import re

# Read the file
with open('wgc_mother_cohort_lollipop.py', 'r') as f:
    content = f.read()

# Replace both instances of the function call
# Pattern: create_lollipop_plot(pct_data, output_plot_path, title, figsize)
# Replace with: create_lollipop_plot(pct_data, output_plot_path, title, figsize, preserve_order)

pattern = r'create_lollipop_plot\(pct_data, output_plot_path, title, figsize\)'
replacement = 'create_lollipop_plot(pct_data, output_plot_path, title, figsize, preserve_order)'

new_content = re.sub(pattern, replacement, content)

# Write back to file
with open('wgc_mother_cohort_lollipop.py', 'w') as f:
    f.write(new_content)

print("Fixed both function calls")