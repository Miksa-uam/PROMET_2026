# Implementation Plan

- [x] 1. Create cluster configuration file





  - Create `scripts/cluster_config.json` with cluster labels and colors
  - Use the structure: `{"cluster_labels": {...}, "cluster_colors": {...}}`
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Add cluster configuration utilities to descriptive_visualizations.py






- [x] 2.1 Add load_cluster_config() function

  - Implement function to load cluster labels and colors from JSON
  - Handle missing file gracefully with warning message and default values
  - Return dictionary with 'cluster_labels' and 'cluster_colors' keys
  - _Requirements: 1.1, 1.2, 1.3, 8.2, 8.3_


- [x] 2.2 Add get_cluster_label() helper function





  - Implement function to get human-readable label for a cluster ID
  - Fall back to "Cluster {id}" format if label not found in config
  - _Requirements: 1.4, 9.4_


- [x] 2.3 Add get_cluster_color() helper function





  - Implement function to get color for a cluster ID
  - Fall back to default palette if color not found in config
  - _Requirements: 1.4, 9.4_

- [x] 3. Extend existing plot functions for cluster support





- [x] 3.1 Modify plot_distribution_comparison() for clusters


  - Add optional `cluster_config_path` parameter
  - Load cluster config when parameter is provided
  - Use cluster labels for x-axis tick labels
  - Use cluster colors for violin plot colors
  - Add `plt.show()` before `plt.close()` to display in notebook
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 9.1, 9.2, 9.3_


- [x] 3.2 Modify plot_stacked_bar_comparison() for clusters

  - Add optional `cluster_config_path` parameter
  - Use cluster labels for x-axis
  - Add `plt.show()` before `plt.close()` to display in notebook
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 9.1, 9.2, 9.3_

- [x] 3.3 Modify plot_multi_lollipop() for clusters


  - Add optional `cluster_config_path` parameter
  - Use cluster labels in legend
  - Use cluster colors for lollipop markers and lines
  - Add `plt.show()` before `plt.close()` to display in notebook
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.2, 9.1, 9.2, 9.3_

- [x] 4. Enhance plot_forest() with secondary y-axis






- [x] 4.1 Add secondary y-axis to forest plot

  - Add optional `cluster_config_path` parameter for cluster label support
  - Add optional `effect_type` parameter ('RD' or 'RR')
  - Create secondary y-axis on the right side
  - Display effect sizes and 95% CIs on secondary axis
  - Format as "RR: X.XX [X.XX-X.XX]" or "RD: X.X% [X.X-X.X%]"
  - Remove significance asterisks (not needed for forest plots)
  - Set reference line at x=1 for RR or x=0 for RD
  - Add `plt.show()` before `plt.close()` to display in notebook
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 9.1, 9.2, 9.3_

- [x] 5. Implement new heatmap function




- [x] 5.1 Create plot_wgc_cluster_heatmap() function


  - Accept prevalence_df with columns: wgc_variable, cluster_id, prevalence_%, n
  - Load cluster config and variable name map
  - Pivot data to matrix format (rows=WGCs, columns=clusters)
  - Create annotations in "n (%)** " format
  - Use YlOrRd colormap with range 0-100%
  - Add significance markers (* or **) based on p-values
  - Save as high-resolution PNG (300 dpi)
  - Add `plt.show()` before `plt.close()` to display in notebook
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 9.1, 9.2, 9.3_

- [x] 6. Add cluster analysis function to descriptive_comparisons.py




- [x] 6.1 Implement cluster_vs_population_mean_analysis() function


  - Follow the same pattern as wgc_vs_population_mean_analysis()
  - Accept df with cluster_id column and WGC binary variables
  - Iterate over clusters (not WGCs) as groups
  - For each cluster, calculate prevalence of each WGC variable
  - Compare each cluster's WGC prevalence to population mean
  - Apply FDR correction if enabled in config
  - Save detailed table with p-values to database
  - Create publication-ready table with asterisks using switch_pvalues_to_asterisks()
  - Return DataFrame for heatmap generation
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.1, 8.2, 8.5_

- [x] 7. Update configuration dataclass




- [x] 7.1 Add cluster_vs_mean_output_table field to descriptive_comparisons_config


  - Add optional field for cluster vs population mean analysis output table name
  - Follow the same pattern as wgc_vs_mean_output_table
  - _Requirements: 10.1, 10.6_

- [x] 8. Add validation and error handling





- [x] 8.1 Add column validation to visualization functions


  - Check that required columns exist in DataFrame before plotting
  - Raise descriptive ValueError if columns are missing
  - _Requirements: 8.1, 8.2, 8.5_

- [x] 8.2 Add empty data checks to visualization functions


  - Check if DataFrame is empty after dropping missing values
  - Print warning and skip plot if no data available
  - _Requirements: 8.3, 8.4_

- [x] 8.3 Add cluster config validation


  - Check that all cluster IDs in data have corresponding labels/colors
  - Print warning for missing cluster IDs and use defaults
  - _Requirements: 1.4, 8.2, 8.3, 8.4_

- [x] 8.4 Add validation to cluster_vs_population_mean_analysis()


  - Check that cluster_col exists in DataFrame
  - Check for sufficient sample sizes in each cluster
  - Print warnings for small clusters
  - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [x] 9. Create example notebook cells




- [x] 9.1 Add notebook cell for cluster_vs_population_mean_analysis


  - Show how to configure and run the analysis
  - Show how to extract results from database
  - _Requirements: 10.6_



- [x] 9.2 Add notebook cell for cluster visualizations
  - Show how to generate violin, stacked bar, and lollipop plots with cluster data


  - Include examples of passing cluster_config_path parameter
  - _Requirements: 10.6_


- [x] 9.3 Add notebook cell for heatmap generation
  - Show how to transform cluster_vs_population_mean_analysis results into heatmap format
  - Show how to extract significance data
  - Show how to call plot_wgc_cluster_heatmap()
  - _Requirements: 10.6_

- [x] 10. Test and validate implementation







- [ ] 10.1 Test cluster config loading
  - Test with valid cluster_config.json
  - Test with missing file (should use defaults)
  - Test with invalid JSON (should handle gracefully)


  - _Requirements: 1.3, 8.2, 8.3_

- [ ] 10.2 Test visualization functions with cluster data
  - Generate violin plot with cluster data
  - Generate stacked bar plot with cluster data
  - Generate lollipop plot with cluster data


  - Verify cluster labels and colors are applied correctly
  - Verify figures display in notebook
  - _Requirements: 2.4, 2.5, 7.1, 7.4, 9.5_

- [x] 10.3 Test forest plot enhancements


  - Generate forest plot with secondary y-axis
  - Verify effect sizes and CIs display correctly
  - Test with both RR and RD effect types
  - Verify no significance asterisks appear
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_



- [ ] 10.4 Test cluster_vs_population_mean_analysis
  - Run analysis with cluster data
  - Verify output table structure matches expected format
  - Verify FDR correction is applied correctly
  - Verify publication-ready table has asterisks


  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 10.5 Test heatmap generation
  - Generate heatmap from cluster_vs_population_mean_analysis results
  - Verify cell annotations show "n (%)** " format
  - Verify colors represent percentages correctly
  - Verify significance markers appear correctly
  - Verify cluster labels and WGC labels are human-readable
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 10.6 Test backward compatibility
  - Run existing WGC analysis code without modifications
  - Verify all existing plots still work
  - Verify no breaking changes to existing functions
  - _Requirements: 10.4, 10.5, 10.6_
