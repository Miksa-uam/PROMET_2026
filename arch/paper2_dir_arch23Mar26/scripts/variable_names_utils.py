#!/usr/bin/env python3
"""
Utility functions for handling human-readable variable names.

This module provides centralized access to the human-readable variable names
dictionary, ensuring consistent naming across all scripts in the project.
"""

import json
import os
from typing import Dict, Optional

# Cache for the loaded dictionary to avoid repeated file reads
_variable_names_cache: Optional[Dict[str, str]] = None

def load_variable_names() -> Dict[str, str]:
    """
    Load the human-readable variable names dictionary from JSON file.
    
    Returns:
        Dict[str, str]: Mapping from snake_case variable names to human-readable names
    """
    global _variable_names_cache
    
    if _variable_names_cache is not None:
        return _variable_names_cache
    
    # Get the path to the JSON file (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "human_readable_variable_names.json")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Filter out comment entries (keys starting with '_comment_')
        _variable_names_cache = {
            key: value for key, value in all_data.items() 
            if not key.startswith('_comment_')
        }
        
        return _variable_names_cache
        
    except FileNotFoundError:
        print(f"Warning: Variable names file not found at {json_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in variable names file: {e}")
        return {}

def get_human_readable_name(variable_name: str) -> str:
    """
    Get the human-readable name for a variable.
    
    Args:
        variable_name (str): The snake_case variable name
        
    Returns:
        str: Human-readable name, or the original variable name if not found
    """
    variable_names = load_variable_names()
    return variable_names.get(variable_name, variable_name)

def get_all_variable_names() -> Dict[str, str]:
    """
    Get all variable name mappings.
    
    Returns:
        Dict[str, str]: Complete mapping from snake_case to human-readable names
    """
    return load_variable_names()

def variable_exists(variable_name: str) -> bool:
    """
    Check if a variable name exists in the dictionary.
    
    Args:
        variable_name (str): The snake_case variable name to check
        
    Returns:
        bool: True if the variable exists in the dictionary
    """
    variable_names = load_variable_names()
    return variable_name in variable_names

def print_all_variables():
    """Print all available variable mappings for debugging."""
    variable_names = load_variable_names()
    print("Available variable name mappings:")
    print("-" * 50)
    for snake_name, human_name in sorted(variable_names.items()):
        print(f"{snake_name:<35} -> {human_name}")
    print(f"\nTotal variables: {len(variable_names)}")

if __name__ == "__main__":
    # Test the utility functions
    print("Testing variable names utility...")
    print_all_variables()
    
    # Test specific lookups
    test_vars = ["age", "baseline_bmi", "womens_health_and_pregnancy", "nonexistent_var"]
    print("\nTesting specific lookups:")
    for var in test_vars:
        human_name = get_human_readable_name(var)
        exists = variable_exists(var)
        print(f"{var:<35} -> {human_name:<40} (exists: {exists})")