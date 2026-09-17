# Implementation Plan

- [x] 1. Create directory structure and file inventory







  - Create the three main directories (scripts/, dbs/, outputs/)
  - Generate complete inventory of current files and their types
  - Create backup of current directory structure
  - _Requirements: 1.1, 2.1, 3.1_

- [x] 2. Implement file movement utilities






  - [x] 2.1 Create file categorization function











    - Write function to identify file types (scripts, databases, outputs)
    - Implement logic to determine destination directory for each file
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2_

  - [x] 2.2 Implement safe file moving functions






    - Write function to copy files to new locations (keeping originals as backup)
    - Add file integrity verification after copying
    - Implement directory creation and file moving orchestration
    - _Requirements: 5.4_

  - [ ]* 2.3 Write unit tests for file movement utilities
    - Test file categorization logic with sample files
    - Test safe copying functionality
    - _Requirements: 5.4_

- [x] 3. Execute file reorganization



  - [x] 3.1 Create directory structure and move files


    - Create scripts/, dbs/, outputs/ directories
    - Move all Python files (.py) and notebooks (.ipynb) to scripts/
    - Move all SQLite databases (.sqlite) to dbs/
    - Move all output directories (pam_clust/, rf_outputs/, etc.) to outputs/
    - Move standalone result files (.png, .svg) to outputs/
    - Verify all files copied successfully with integrity checks
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2, 3.3_

- [x] 4. Update database connection paths



  - [x] 4.1 Update database paths in Python configuration files


    - Update rf_config.py and paper12_config.py to use "../dbs/" prefix
    - Change absolute paths to relative paths for database connections
    - _Requirements: 2.2, 5.2_

  - [x] 4.2 Update database paths in notebooks



    - Update all database references in paper2_notebook_kiro_refact.ipynb
    - Change absolute paths to relative paths from scripts/ directory
    - _Requirements: 2.2, 5.2_

  - [ ]* 4.3 Test database connections
    - Write test script to verify all database connections work
    - Test from scripts/ directory context
    - _Requirements: 5.2_

- [x] 5. Update output generation paths





  - [x] 5.1 Update output paths in configuration files

    - Update rf_config.py and paper12_config.py output_dir to use "../outputs/" prefix
    - Update rf_engine.py to handle relative output paths correctly

    - _Requirements: 3.4, 5.3_

  - [x] 5.2 Update output paths in notebooks

    - Update all output generation code in notebooks to use "../outputs/" prefix
    - Ensure outputs are written to correct subdirectories
    - _Requirements: 3.4, 5.3_

  - [ ]* 5.3 Test output generation
    - Run sample output generation to verify paths work
    - Confirm files are created in expected locations
    - _Requirements: 5.3_

- [x] 6. Validate and test reorganized structure



  - [x] 6.1 Test Python module imports and notebook execution


    - Verify all Python modules can import each other from scripts/ directory
    - Execute key notebook cells to ensure they run without errors
    - Test database connections and output generation
    - _Requirements: 1.4, 1.5, 4.2, 4.3, 5.1, 5.5_

  - [x] 6.2 Create documentation and cleanup


    - Write summary of new directory structure in README or documentation
    - Document import patterns and relative path usage
    - Remove original files from root directory after verification
    - _Requirements: 4.4, 5.4_

  - [ ]* 6.3 Create migration verification report
    - Generate report showing all files moved successfully
    - Document any issues encountered and resolutions
  