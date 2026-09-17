"""
Script to implement Task 1: Create directory structure and file inventory.

This script:
1. Creates the three main directories (scripts/, dbs/, outputs/)
2. Generates complete inventory of current files and their types
3. Creates backup of current directory structure
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from file_movement_utils import get_file_mapping, categorize_directory_contents, categorize_subdirectories


def create_main_directories():
    """Create the three main directories for the reorganized structure."""
    directories = ["scripts", "dbs", "outputs"]
    created_dirs = []
    
    print("Creating main directory structure...")
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            created_dirs.append(directory)
            print(f"✓ Created directory: {directory}/")
        except OSError as e:
            print(f"✗ Error creating directory {directory}: {e}")
            return False, created_dirs
    
    return True, created_dirs


def generate_file_inventory():
    """Generate a complete inventory of current files and their categorization."""
    print("\nGenerating file inventory...")
    
    # Get file categorization
    categorized_files = categorize_directory_contents()
    output_dirs = categorize_subdirectories()
    file_mapping = get_file_mapping()
    
    # Create inventory structure
    inventory = {
        "timestamp": datetime.now().isoformat(),
        "total_files": 0,
        "total_directories": len(output_dirs),
        "categorized_files": {},
        "output_directories": output_dirs,
        "complete_file_mapping": {}
    }
    
    # Process categorized files
    for file_type, file_list in categorized_files.items():
        inventory["categorized_files"][file_type.value] = {
            "count": len(file_list),
            "files": file_list
        }
        inventory["total_files"] += len(file_list)
    
    # Process file mapping
    for original_path, (file_type, destination_path) in file_mapping.items():
        inventory["complete_file_mapping"][original_path] = {
            "type": file_type.value,
            "destination": destination_path,
            "is_directory": Path(original_path).is_dir()
        }
    
    # Save inventory to JSON file
    inventory_file = "project_inventory.json"
    try:
        with open(inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)
        print(f"✓ File inventory saved to: {inventory_file}")
        
        # Print summary
        print(f"\nInventory Summary:")
        print(f"  Total files: {inventory['total_files']}")
        print(f"  Total directories: {inventory['total_directories']}")
        print(f"  Script files: {inventory['categorized_files']['script']['count']}")
        print(f"  Database files: {inventory['categorized_files']['database']['count']}")
        print(f"  Output files: {inventory['categorized_files']['output']['count']}")
        print(f"  Unknown files: {inventory['categorized_files']['unknown']['count']}")
        
        return True, inventory
        
    except IOError as e:
        print(f"✗ Error saving inventory: {e}")
        return False, None


def create_directory_backup():
    """Create a backup of the current directory structure."""
    print("\nCreating backup of current directory structure...")
    
    backup_name = f"backup_project_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create backup directory
        backup_path = Path(backup_name)
        backup_path.mkdir(exist_ok=True)
        
        # Get list of items to backup (exclude the backup directory itself and .kiro)
        items_to_backup = []
        for item in Path(".").iterdir():
            if (item.name != backup_name and 
                not item.name.startswith('.') and 
                item.name != '__pycache__'):
                items_to_backup.append(item)
        
        # Copy each item to backup
        backed_up_items = []
        for item in items_to_backup:
            try:
                if item.is_file():
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, backup_path / item.name)
                backed_up_items.append(str(item))
                print(f"  ✓ Backed up: {item}")
            except (OSError, IOError) as e:
                print(f"  ✗ Error backing up {item}: {e}")
        
        # Create backup manifest
        backup_manifest = {
            "backup_timestamp": datetime.now().isoformat(),
            "backup_directory": backup_name,
            "backed_up_items": backed_up_items,
            "total_items": len(backed_up_items)
        }
        
        manifest_file = backup_path / "backup_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(backup_manifest, f, indent=2)
        
        print(f"✓ Backup created successfully: {backup_name}/")
        print(f"✓ Backup manifest: {manifest_file}")
        print(f"  Total items backed up: {len(backed_up_items)}")
        
        return True, backup_name
        
    except (OSError, IOError) as e:
        print(f"✗ Error creating backup: {e}")
        return False, None


def main():
    """Execute Task 1: Create directory structure and file inventory."""
    print("=" * 60)
    print("TASK 1: Create directory structure and file inventory")
    print("=" * 60)
    
    success_count = 0
    total_tasks = 3
    
    # Step 1: Create main directories
    dirs_success, created_dirs = create_main_directories()
    if dirs_success:
        success_count += 1
    
    # Step 2: Generate file inventory
    inventory_success, inventory = generate_file_inventory()
    if inventory_success:
        success_count += 1
    
    # Step 3: Create backup
    backup_success, backup_name = create_directory_backup()
    if backup_success:
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETION SUMMARY")
    print("=" * 60)
    print(f"Tasks completed successfully: {success_count}/{total_tasks}")
    
    if dirs_success:
        print(f"✓ Directories created: {', '.join(created_dirs)}")
    else:
        print("✗ Directory creation failed")
    
    if inventory_success:
        print("✓ File inventory generated: project_inventory.json")
    else:
        print("✗ File inventory generation failed")
    
    if backup_success:
        print(f"✓ Backup created: {backup_name}/")
    else:
        print("✗ Backup creation failed")
    
    if success_count == total_tasks:
        print("\n🎉 Task 1 completed successfully!")
        print("Ready to proceed with Task 2.2: Implement safe file moving functions")
    else:
        print(f"\n⚠️  Task 1 partially completed ({success_count}/{total_tasks} steps)")
        print("Please resolve any errors before proceeding to the next task.")
    
    return success_count == total_tasks


if __name__ == "__main__":
    main()