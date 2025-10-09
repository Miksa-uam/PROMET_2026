# Requirements Document

## Introduction

This feature involves reorganizing the paper2_dir research project into a cleaner, more maintainable structure. The current directory has mixed file types at the root level, making it difficult to navigate and maintain. The reorganization will create a simple 3-folder structure (scripts/, dbs/, outputs/) that separates code, data, and results while maintaining all existing functionality.

## Requirements

### Requirement 1

**User Story:** As a researcher working on the weight gain causes analysis project, I want all my code files organized in a single scripts/ directory, so that I can easily find and manage all executable components in one place.

#### Acceptance Criteria

1. WHEN the reorganization is complete THEN the scripts/ directory SHALL contain all Python modules (.py files)
2. WHEN the reorganization is complete THEN the scripts/ directory SHALL contain all Jupyter notebooks (.ipynb files)  
3. WHEN the reorganization is complete THEN the scripts/ directory SHALL contain all configuration files (.py config files)
4. WHEN a Python module is moved to scripts/ THEN all import statements in other files SHALL be updated to reflect the new path
5. WHEN a notebook references a Python module THEN the import paths SHALL be updated to work from the new structure

### Requirement 2

**User Story:** As a researcher managing multiple database files, I want all SQLite databases organized in a dedicated dbs/ directory, so that I can easily locate and manage my data files separately from code.

#### Acceptance Criteria

1. WHEN the reorganization is complete THEN the dbs/ directory SHALL contain all SQLite database files (.sqlite files)
2. WHEN database files are moved THEN all database connection strings in code SHALL be updated to reflect new paths
3. WHEN the reorganization is complete THEN no database files SHALL remain in the root directory

### Requirement 3

**User Story:** As a researcher generating various analysis outputs, I want all results organized in an outputs/ directory with logical subfolder grouping, so that I can easily find results from specific experiments or analyses.

#### Acceptance Criteria

1. WHEN the reorganization is complete THEN the outputs/ directory SHALL contain all existing result subdirectories (pam_clust/, rf_outputs/, etc.)
2. WHEN the reorganization is complete THEN the outputs/ directory SHALL contain all standalone result files (PNG, SVG files)
3. WHEN the reorganization is complete THEN output subdirectories SHALL maintain their internal structure
4. WHEN code generates new outputs THEN the output paths SHALL be updated to write to the outputs/ directory
5. IF temporary files exist THEN they SHALL be moved to outputs/ or cleaned up as appropriate

### Requirement 4

**User Story:** As a novice software engineer, I want the import path changes to be clearly documented and tested, so that I can understand how the new structure affects code execution and learn proper Python import practices.

#### Acceptance Criteria

1. WHEN import paths are updated THEN the changes SHALL use relative imports where appropriate (e.g., from scripts.module_name import function)
2. WHEN the reorganization is complete THEN all notebooks SHALL run successfully with updated import paths
3. WHEN the reorganization is complete THEN all Python modules SHALL import successfully from their new locations
4. WHEN import changes are made THEN a summary document SHALL be created explaining the new import patterns
5. IF any imports fail after reorganization THEN clear error messages and solutions SHALL be provided

### Requirement 5

**User Story:** As a researcher who needs to maintain project functionality, I want the reorganization to preserve all existing capabilities, so that my analysis pipeline continues to work without interruption.

#### Acceptance Criteria

1. WHEN the reorganization is complete THEN all existing functionality SHALL remain intact
2. WHEN the reorganization is complete THEN all database connections SHALL work with updated paths
3. WHEN the reorganization is complete THEN all output generation SHALL work with updated paths
4. WHEN the reorganization is complete THEN no files SHALL be lost or corrupted during the move
5. WHEN testing the reorganized structure THEN all critical workflows SHALL execute successfully