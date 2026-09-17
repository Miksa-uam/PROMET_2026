# Project Reorganization Documentation

## Overview

This project has been reorganized into a cleaner, more maintainable directory structure. All files have been categorized and moved to appropriate directories with updated path references.

## New Directory Structure

```
project_root/
├── scripts/                    # All Python scripts and Jupyter notebooks
│   ├── *.py                   # Python modules and analysis scripts
│   └── *.ipynb               # Jupyter notebooks
├── dbs/                       # All database files
│   ├── *.sqlite              # SQLite database files
│   └── pnk_db2_*             # Project-specific databases
├── outputs/                   # All generated outputs and results
│   ├── rf_outputs/           # Random Forest analysis outputs
│   ├── wgc_association_networks/  # Network analysis outputs
│   ├── hclust_dendrograms/   # Hierarchical clustering dendrograms
│   ├── pam_clust/           # PAM clustering results
│   ├── arch/                # Architecture-related outputs
│   ├── clustering_metadata/ # Clustering metadata
│   └── explorative_analyses_Aug25/  # Exploratory analysis results
└── .kiro/                    # Kiro IDE configuration and specs
```

## Key Changes Made

### 1. File Organization
- **Scripts**: All `.py` files and `.ipynb` notebooks moved to `scripts/`
- **Databases**: All `.sqlite` files moved to `dbs/`
- **Outputs**: All result directories and output files moved to `outputs/`

### 2. Path Updates
All hardcoded absolute paths have been updated to relative paths:

#### Database Paths
- **Before**: `C:/Users/.../paper2_dir/pnk_db2_p2_in.sqlite`
- **After**: `../dbs/pnk_db2_p2_in.sqlite` (from scripts directory)

#### Output Paths
- **Before**: `C:/Users/.../paper2_dir/rf_outputs`
- **After**: `../outputs/rf_outputs` (from scripts directory)

### 3. Configuration Updates
- `scripts/rf_config.py`: Updated database and output paths
- `scripts/paper12_config.py`: Updated database and output paths
- `scripts/paper2_notebook_kiro_refact.ipynb`: Updated all path references

## Working with the New Structure

### Running Scripts
All scripts should be run from the `scripts/` directory:

```bash
cd scripts/
python rf_engine.py
python descriptive_comparisons.py
```

### Running Notebooks
Jupyter notebooks are located in `scripts/` and use relative paths:

```bash
cd scripts/
jupyter notebook paper2_notebook_kiro_refact.ipynb
```

### Import Patterns
Python modules can import each other directly when running from the `scripts/` directory:

```python
from paper12_config import paper2_rf_config
from rf_engine import RandomForestAnalyzer
```

### Database Access
Databases are accessed using relative paths from the `scripts/` directory:

```python
import sqlite3
conn = sqlite3.connect("../dbs/pnk_db2_p2_in.sqlite")
```

### Output Generation
All outputs are written to the `outputs/` directory:

```python
import os
output_path = "../outputs/rf_outputs/my_analysis.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
```

## Validation

The reorganized structure has been validated with comprehensive tests:

✅ **Module Imports**: All Python modules can be imported correctly  
✅ **Database Connections**: All databases are accessible with new paths  
✅ **Output Directories**: All output directories exist and are writable  
✅ **Notebook Functionality**: Key notebook operations work correctly  

## Benefits of the New Structure

1. **Clarity**: Clear separation of code, data, and outputs
2. **Portability**: Relative paths make the project portable across systems
3. **Maintainability**: Easier to understand and maintain the codebase
4. **Scalability**: Better organization for future development
5. **Collaboration**: Cleaner structure for team collaboration

## Migration Notes

- Original files remain in the root directory as backups
- All functionality has been preserved with updated paths
- No data or analysis results were lost during reorganization
- The reorganization is fully backward compatible

## Troubleshooting

If you encounter path-related issues:

1. Ensure you're running scripts from the `scripts/` directory
2. Verify that relative paths use `../` to go up one directory level
3. Check that all required directories exist in `outputs/`
4. Confirm database files exist in `dbs/`

For any issues, refer to the validation tests in `test_imports.py` and `test_notebook_execution.py`.