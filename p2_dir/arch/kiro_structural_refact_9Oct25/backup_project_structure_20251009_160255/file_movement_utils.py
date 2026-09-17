"""
File movement utilities for project reorganization.

This module provides functions to categorize files and safely move them
to the new directory structure (scripts/, dbs/, outputs/).
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging


class FileType(Enum):
    """Enumeration of file types for categorization."""
    SCRIPT = "script"
    DATABASE = "database"
    OUTPUT = "output"
    UNKNOWN = "unknown"


def get_file_category(file_path: str) -> FileType:
    """
    Categorize a file based on its extension and name patterns.
    
    Args:
        file_path (str): Path to the file to categorize
        
    Returns:
        FileType: The category of the file (SCRIPT, DATABASE, OUTPUT, or UNKNOWN)
    """
    file_path = Path(file_path)
    file_name = file_path.name.lower()
    file_extension = file_path.suffix.lower()
    
    # Script files - Python modules, notebooks, and config files
    if file_extension in ['.py', '.ipynb']:
        return FileType.SCRIPT
    
    # Database files - SQLite databases
    if file_extension == '.sqlite' or file_name.endswith('.sqlite'):
        return FileType.DATABASE
    
    # Output files - images, results, and analysis outputs
    if file_extension in ['.png', '.svg', '.jpg', '.jpeg', '.pdf', '.xlsx', '.csv']:
        return FileType.OUTPUT
    
    # Special cases for files without extensions that are databases
    if any(db_pattern in file_name for db_pattern in ['pnk_db', 'cluster', 'goldstd']):
        return FileType.DATABASE
    
    return FileType.UNKNOWN


def determine_destination_directory(file_path: str, file_type: FileType) -> str:
    """
    Determine the destination directory for a file based on its type.
    
    Args:
        file_path (str): Original path of the file
        file_type (FileType): Category of the file
        
    Returns:
        str: Destination directory path
    """
    file_name = Path(file_path).name
    
    if file_type == FileType.SCRIPT:
        return os.path.join("scripts", file_name)
    elif file_type == FileType.DATABASE:
        return os.path.join("dbs", file_name)
    elif file_type == FileType.OUTPUT:
        return os.path.join("outputs", file_name)
    else:
        # Keep unknown files in root or handle specially
        return file_name


def categorize_directory_contents(directory_path: str = ".") -> Dict[FileType, List[str]]:
    """
    Categorize all files in a directory by their type.
    
    Args:
        directory_path (str): Path to directory to analyze (default: current directory)
        
    Returns:
        Dict[FileType, List[str]]: Dictionary mapping file types to lists of file paths
    """
    categorized_files = {
        FileType.SCRIPT: [],
        FileType.DATABASE: [],
        FileType.OUTPUT: [],
        FileType.UNKNOWN: []
    }
    
    # Get all files in the directory (not subdirectories)
    directory = Path(directory_path)
    
    for item in directory.iterdir():
        if item.is_file():
            # Skip hidden files and system files
            if not item.name.startswith('.') and item.name != 'desktop.ini':
                file_type = get_file_category(str(item))
                categorized_files[file_type].append(str(item))
    
    return categorized_files


def categorize_subdirectories(directory_path: str = ".") -> List[str]:
    """
    Identify subdirectories that should be moved to outputs/.
    
    Args:
        directory_path (str): Path to directory to analyze (default: current directory)
        
    Returns:
        List[str]: List of subdirectory paths that should be moved to outputs/
    """
    output_directories = []
    directory = Path(directory_path)
    
    # Directories that should be moved to outputs/ based on the design document
    output_dir_patterns = [
        'pam_clust', 'rf_outputs', 'wgc_association_networks', 
        'arch', 'clustering_metadata', 'explorative_analyses_Aug25'
    ]
    
    for item in directory.iterdir():
        if item.is_dir():
            # Skip system and hidden directories
            if not item.name.startswith('.') and item.name != '__pycache__':
                if item.name in output_dir_patterns:
                    output_directories.append(str(item))
    
    return output_directories


def get_file_mapping(directory_path: str = ".") -> Dict[str, Tuple[FileType, str]]:
    """
    Create a complete mapping of files to their destination paths.
    
    Args:
        directory_path (str): Path to directory to analyze (default: current directory)
        
    Returns:
        Dict[str, Tuple[FileType, str]]: Mapping of original path to (file_type, destination_path)
    """
    file_mapping = {}
    
    # Categorize individual files
    categorized_files = categorize_directory_contents(directory_path)
    
    for file_type, file_list in categorized_files.items():
        for file_path in file_list:
            destination = determine_destination_directory(file_path, file_type)
            file_mapping[file_path] = (file_type, destination)
    
    # Add subdirectories that should go to outputs/
    output_dirs = categorize_subdirectories(directory_path)
    for dir_path in output_dirs:
        dir_name = Path(dir_path).name
        destination = os.path.join("outputs", dir_name)
        file_mapping[dir_path] = (FileType.OUTPUT, destination)
    
    return file_mapping


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file for integrity verification.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: SHA-256 hash of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def verify_file_integrity(source_path: str, destination_path: str) -> bool:
    """
    Verify that a copied file has the same content as the original.
    
    Args:
        source_path (str): Path to the original file
        destination_path (str): Path to the copied file
        
    Returns:
        bool: True if files are identical, False otherwise
    """
    try:
        source_hash = calculate_file_hash(source_path)
        dest_hash = calculate_file_hash(destination_path)
        return source_hash == dest_hash
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Error verifying file integrity: {e}")
        return False


def create_directory_structure() -> bool:
    """
    Create the main directory structure (scripts/, dbs/, outputs/).
    
    Returns:
        bool: True if directories were created successfully, False otherwise
    """
    directories = ["scripts", "dbs", "outputs"]
    
    try:
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logging.info(f"Created directory: {directory}")
        return True
    except OSError as e:
        logging.error(f"Error creating directory structure: {e}")
        return False


def safe_copy_file(source_path: str, destination_path: str) -> bool:
    """
    Safely copy a file to a new location with integrity verification.
    
    Args:
        source_path (str): Path to the source file
        destination_path (str): Path where the file should be copied
        
    Returns:
        bool: True if copy was successful and verified, False otherwise
    """
    try:
        # Ensure destination directory exists
        dest_dir = Path(destination_path).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(source_path, destination_path)
        logging.info(f"Copied {source_path} to {destination_path}")
        
        # Verify integrity
        if verify_file_integrity(source_path, destination_path):
            logging.info(f"File integrity verified for {destination_path}")
            return True
        else:
            logging.error(f"File integrity check failed for {destination_path}")
            # Remove the corrupted copy
            try:
                os.remove(destination_path)
                logging.info(f"Removed corrupted copy: {destination_path}")
            except OSError:
                pass
            return False
            
    except (OSError, IOError) as e:
        logging.error(f"Error copying file {source_path} to {destination_path}: {e}")
        return False


def safe_copy_directory(source_path: str, destination_path: str) -> bool:
    """
    Safely copy a directory to a new location with integrity verification.
    
    Args:
        source_path (str): Path to the source directory
        destination_path (str): Path where the directory should be copied
        
    Returns:
        bool: True if copy was successful and verified, False otherwise
    """
    try:
        # Copy the entire directory tree
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        logging.info(f"Copied directory {source_path} to {destination_path}")
        
        # Verify integrity of all files in the directory
        source_dir = Path(source_path)
        dest_dir = Path(destination_path)
        
        for source_file in source_dir.rglob("*"):
            if source_file.is_file():
                # Calculate relative path and corresponding destination file
                relative_path = source_file.relative_to(source_dir)
                dest_file = dest_dir / relative_path
                
                if not verify_file_integrity(str(source_file), str(dest_file)):
                    logging.error(f"Directory copy integrity check failed for {dest_file}")
                    # Remove the entire corrupted directory copy
                    try:
                        shutil.rmtree(destination_path)
                        logging.info(f"Removed corrupted directory copy: {destination_path}")
                    except OSError:
                        pass
                    return False
        
        logging.info(f"Directory integrity verified for {destination_path}")
        return True
        
    except (OSError, IOError) as e:
        logging.error(f"Error copying directory {source_path} to {destination_path}: {e}")
        return False


def execute_file_reorganization(dry_run: bool = True) -> Dict[str, bool]:
    """
    Execute the complete file reorganization process.
    
    Args:
        dry_run (bool): If True, only simulate the moves without actually copying files
        
    Returns:
        Dict[str, bool]: Dictionary mapping file paths to success status
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    results = {}
    
    # Create directory structure first
    if not dry_run:
        if not create_directory_structure():
            logging.error("Failed to create directory structure")
            return results
    else:
        logging.info("DRY RUN: Would create directories: scripts/, dbs/, outputs/")
    
    # Get file mapping
    file_mapping = get_file_mapping()
    
    # Process each file/directory
    for source_path, (file_type, destination_path) in file_mapping.items():
        if dry_run:
            logging.info(f"DRY RUN: Would move {source_path} to {destination_path}")
            results[source_path] = True
        else:
            # Check if source is a file or directory
            source = Path(source_path)
            
            if source.is_file():
                success = safe_copy_file(source_path, destination_path)
            elif source.is_dir():
                success = safe_copy_directory(source_path, destination_path)
            else:
                logging.warning(f"Source path does not exist: {source_path}")
                success = False
            
            results[source_path] = success
            
            if success:
                logging.info(f"Successfully processed: {source_path} -> {destination_path}")
            else:
                logging.error(f"Failed to process: {source_path} -> {destination_path}")
    
    # Summary
    successful_moves = sum(1 for success in results.values() if success)
    total_moves = len(results)
    
    if dry_run:
        logging.info(f"DRY RUN COMPLETE: Would process {total_moves} items")
    else:
        logging.info(f"REORGANIZATION COMPLETE: {successful_moves}/{total_moves} items processed successfully")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        # Execute actual file reorganization
        print("EXECUTING FILE REORGANIZATION...")
        print("=" * 50)
        results = execute_file_reorganization(dry_run=False)
        
        # Print results summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        print(f"\nResults: {successful}/{total} items processed successfully")
        
        if successful < total:
            print("\nFailed items:")
            for path, success in results.items():
                if not success:
                    print(f"  - {path}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        # Execute dry run
        print("DRY RUN - File reorganization simulation:")
        print("=" * 50)
        results = execute_file_reorganization(dry_run=True)
    
    else:
        # Test the categorization functions
        print("File categorization test:")
        print("=" * 50)
        
        # Test individual file categorization
        test_files = [
            "descriptive_comparisons.py",
            "paper2_notebook.ipynb", 
            "pnk_db2_p2_in.sqlite",
            "wgc_cmpl_datacollection_bias.png",
            "paper12_config.py"
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                file_type = get_file_category(test_file)
                destination = determine_destination_directory(test_file, file_type)
                print(f"{test_file} -> {file_type.value} -> {destination}")
        
        print("\nComplete directory categorization:")
        print("=" * 50)
        
        # Test complete directory categorization
        categorized = categorize_directory_contents()
        for file_type, files in categorized.items():
            if files:
                print(f"\n{file_type.value.upper()} files:")
                for file_path in files:
                    print(f"  - {file_path}")
        
        print("\nOutput directories:")
        output_dirs = categorize_subdirectories()
        for dir_path in output_dirs:
            print(f"  - {dir_path}")
        
        print("\nComplete file mapping:")
        print("=" * 50)
        mapping = get_file_mapping()
        for original, (file_type, destination) in mapping.items():
            print(f"{original} -> {destination} ({file_type.value})")
        
        print("\n" + "=" * 50)
        print("Usage:")
        print("  python file_movement_utils.py           # Show file categorization")
        print("  python file_movement_utils.py --dry-run # Simulate reorganization")
        print("  python file_movement_utils.py --execute # Execute reorganization")