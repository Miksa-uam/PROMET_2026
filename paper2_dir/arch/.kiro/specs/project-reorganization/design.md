# Design Document

## Overview

The project reorganization will transform the current flat directory structure into a clean 3-folder hierarchy: scripts/, dbs/, and outputs/. This design prioritizes simplicity and maintainability while preserving all existing functionality. The reorganization will be implemented as a systematic migration process that updates file locations and all corresponding references.

## Architecture

### New Directory Structure
```
paper2_dir/
├── scripts/
│   ├── *.py (all Python modules)
│   ├── *.ipynb (all Jupyter notebooks)
│   └── *config.py (all configuration files)
├── dbs/
│   └── *.sqlite (all database files)
└── outputs/
    ├── pam_clust/
    ├── rf_outputs/
    ├── wgc_association_networks/
    ├── arch/
    ├── clustering_metadata/
    ├── explorative_analyses_Aug25/
    └── *.png, *.svg (standalone result files)
```

### Migration Strategy
The reorganization will follow a safe, incremental approach:
1. **Create new directory structure**
2. **Copy files to new locations** (keeping originals as backup)
3. **Update all import statements and file paths**
4. **Test functionality thoroughly**
5. **Remove original files after verification**

## Components and Interfaces

### File Movement Component
**Purpose:** Safely relocate files to their new directory structure

**Key Functions:**
- `create_directory_structure()` - Creates the three main directories
- `move_scripts()` - Moves all .py, .ipynb, and config files to scripts/
- `move_databases()` - Moves all .sqlite files to dbs/
- `move_outputs()` - Moves result directories and files to outputs/

### Import Path Updater Component
**Purpose:** Update all import statements to work with the new structure

**Key Functions:**
- `scan_import_statements()` - Identifies all import statements in Python files and notebooks
- `update_python_imports()` - Updates import paths in .py files
- `update_notebook_imports()` - Updates import paths in .ipynb files
- `validate_imports()` - Tests that all imports work correctly

### Database Path Updater Component
**Purpose:** Update database connection strings throughout the codebase

**Key Functions:**
- `find_database_references()` - Locates all database file references
- `update_database_paths()` - Updates paths to point to dbs/ directory
- `test_database_connections()` - Verifies all database connections work

### Output Path Updater Component
**Purpose:** Update output file generation paths

**Key Functions:**
- `find_output_references()` - Locates all output file generation code
- `update_output_paths()` - Updates paths to write to outputs/ directory
- `verify_output_generation()` - Tests that outputs are generated correctly

## Data Models

### File Mapping Model
```python
class FileMapping:
    original_path: str
    new_path: str
    file_type: str  # 'script', 'database', 'output'
    dependencies: List[str]  # Files that reference this file
```

### Import Statement Model
```python
class ImportStatement:
    file_path: str
    line_number: int
    original_import: str
    updated_import: str
    import_type: str  # 'absolute', 'relative', 'local'
```

## Error Handling

### File Operation Errors
- **Missing files:** Log warning and continue with available files
- **Permission errors:** Provide clear instructions for resolving access issues
- **Disk space:** Check available space before starting migration

### Import Resolution Errors
- **Circular imports:** Detect and report circular dependencies
- **Missing modules:** Identify and list any modules that cannot be found
- **Syntax errors:** Provide line-by-line error reporting with suggested fixes

### Database Connection Errors
- **Invalid paths:** Automatically attempt common path corrections
- **Locked databases:** Provide instructions for closing database connections
- **Corrupted files:** Verify database integrity before and after move

## Testing Strategy

### Pre-Migration Testing
1. **Inventory all files** - Create complete list of current files and their purposes
2. **Test current functionality** - Run key notebooks and scripts to establish baseline
3. **Backup creation** - Create full backup of current directory structure

### Migration Testing
1. **Incremental verification** - Test each component move individually
2. **Import testing** - Verify each updated import statement works
3. **Database testing** - Test all database connections after path updates
4. **Output testing** - Verify output generation works with new paths

### Post-Migration Testing
1. **Full workflow testing** - Run complete analysis pipeline end-to-end
2. **Notebook execution** - Execute all notebooks to ensure they run without errors
3. **Cross-reference validation** - Verify all file references are correctly updated
4. **Performance verification** - Ensure no performance degradation from new structure

### Import Path Examples

#### Before Reorganization
```python
# In notebook at root level
import descriptive_comparisons
from paper12_config import CONFIG
import stats_helpers as sh
```

#### After Reorganization
```python
# In notebook now in scripts/ directory
import descriptive_comparisons  # Same directory, no change needed
from paper12_config import CONFIG  # Same directory, no change needed
import stats_helpers as sh  # Same directory, no change needed

# For database paths
db_path = "../dbs/pnk_db2_p2_in.sqlite"  # Updated to relative path

# For output paths  
output_path = "../outputs/pam_clust/results.png"  # Updated to relative path
```

#### Cross-Directory Imports (Future Consideration)
```python
# If importing from outside scripts/ directory (not needed for current reorganization)
import sys
sys.path.append('../scripts')
import descriptive_comparisons
```

## Implementation Notes

### Minimal Complexity Approach
- All code files stay together in scripts/ - no subfolders initially
- Import statements between Python files require minimal changes since they'll be in the same directory
- Only file paths (databases, outputs) need updating, not import statements
- Notebooks and Python files can import each other using simple names

### Path Update Patterns
- **Database paths:** Change from `"filename.sqlite"` to `"../dbs/filename.sqlite"`
- **Output paths:** Change from `"output.png"` to `"../outputs/output.png"`
- **Subdirectory outputs:** Change from `"pam_clust/result.txt"` to `"../outputs/pam_clust/result.txt"`

This design ensures the reorganization is straightforward while maintaining all functionality and providing a foundation for future improvements.