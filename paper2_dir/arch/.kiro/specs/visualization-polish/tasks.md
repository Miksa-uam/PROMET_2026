# Implementation Plan

- [x] 1. Enhance table output in analyze_cluster_vs_population function





  - Modify column header generation to use cluster labels from configuration
  - Update first column header to "Whole population: Mean (±SD) / N (%)"
  - Update cluster column headers to "[Cluster Name]: Mean (±SD) / N (%)" format
  - Update p-value column headers to "[Cluster Name]: p-value" format
  - Add cluster_config_path parameter with default value 'cluster_config.json'
  - Display variable names using get_nice_name() for human-readable format
  - Add DataFrame display output with pandas display options configured for wide tables
  - Test with full configuration, missing configuration, and partial configuration
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_



- [x] 2. Adjust lollipop plot significance markers and separators



  - Reduce x_offset for significance markers from 0.02 to 0.01 for closer positioning
  - Increase variable separator linewidth from 0.5 to 2.0 for bold appearance
  - Verify markers align horizontally with lollipop heads
  - Test with multiple variables and single variable scenarios
  - _Requirements: 8.1, 8.2, 8.3, 9.1, 9.2, 9.3_
-

- [x] 3. Reduce forest plot axis spacing





  - Change wspace parameter from 0.05 to 0.01 in gridspec_kw
  - Verify effect size axis appears close to main plot
  - Test with both RR and RD plot types
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 4. Adjust stacked bar plot legend and y-axis





  - Add bbox_to_anchor=(1.0, 1.08) to legend positioning
  - Change y-axis limit from 120 to 100
  - Add clip_on=False to text elements for sample sizes and significance markers
  - Verify legend positioned above plot without overlap
  - Verify y-axis shows 0-100% scale
  - Verify sample sizes and significance markers remain visible
  - _Requirements: 11.1, 11.2, 11.3, 12.1, 12.2, 12.3_

- [x] 5. Adjust violin plot legend and label positioning





  - Add legend positioning with loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2
  - Update x-axis label alignment to ha='center' with rotation=45
  - Verify legend positioned above plot area
  - Verify labels centered on violin distributions
  - _Requirements: 13.1, 13.2, 14.1, 14.2_

- [x] 6. Implement heatmap population column





  - Add helper function _extract_percentage_from_table_cell() to parse table cell values
  - Extract population prevalence data from results_df for each variable
  - Insert population data as first entries in prevalence_data list with cluster_id='Population'
  - Reorder matrix columns to ensure 'Population' appears first
  - Update column label generation to include "Whole population" label
  - Test with various variable counts and verify population column appears first
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 7. Implement heatmap cluster-specific color scales




  - Import LinearSegmentedColormap from matplotlib.colors
  - Replace single seaborn heatmap with multiple subplot approach
  - Create white-to-red colormap for population column
  - Create white-to-[cluster-color] colormaps for each cluster column
  - Use imshow() for each subplot with appropriate colormap and vmin=0, vmax=100
  - Add annotations manually to each subplot matching original format
  - Add row labels to leftmost axis only
  - Add column labels to all axes with centered alignment
  - Remove colorbar from figure
  - Test with full cluster colors configuration and missing colors (fallback)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2_

- [x] 8. Fix heatmap variable ordering





  - Use reindex() on matrix and n_matrix to match input variables list order
  - Iterate over variables list instead of matrix.index when creating annotations
  - Verify rows appear in exact order provided in function call
  - Test with various variable orderings
  - _Requirements: 7.1, 7.2_

- [x] 9. Implement violin plot cluster-specific colors





  - Refactor _plot_single_violin() to use subplot approach with one subplot per cluster
  - Create figure with subplots using plt.subplots(1, n_clusters, sharey=True)
  - For each cluster subplot, prepare separate plot_data with population and cluster data
  - Get cluster-specific color using get_cluster_color()
  - Create palette with POPULATION_COLOR for population side and cluster color for cluster side
  - Plot split violin on each subplot with cluster-specific palette
  - Add significance markers to each subplot
  - Set cluster label as subplot title
  - Add y-axis label to leftmost subplot only
  - Adjust figure size based on number of clusters
  - Test with multiple clusters and verify each uses correct color
  - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 10. Update plot_cluster_heatmap to parse new table column format





  - Update column extraction logic to handle cluster labels instead of numeric IDs
  - Add helper function _parse_cluster_column_header() to extract cluster identifier
  - Update regex patterns to match new header format "[Cluster Name]: Mean (±SD) / N (%)"
  - Add fallback parsing for old format "Cluster X: Mean/N"
  - Test with new table format from updated analyze_cluster_vs_population
  - _Requirements: 1.5_
