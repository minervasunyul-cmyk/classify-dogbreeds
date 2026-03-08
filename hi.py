"""
CLI script to check the time difference between the first and last created file in a directory.
Usage: python check_file_time_diff.py <directory_path>
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def get_file_creation_times(directory):
    """Get creation times of all files in directory."""
    files_with_times = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file():
            try:
                # Get creation time (Windows) or modification time (Unix)
                if os.name == 'nt':  # Windows
                    creation_time = datetime.fromtimestamp(file_path.stat().st_ctime)
                else:  # Unix/Linux/Mac
                    creation_time = datetime.fromtimestamp(file_path.stat().st_birthtime)
                
                files_with_times.append((file_path, creation_time))
            except (OSError, ValueError) as e:
                continue
    
    return files_with_times

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_file_time_diff.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    files_with_times = get_file_creation_times(directory)
    
    if not files_with_times:
        print(f"No files found in '{directory}'")
        sys.exit(1)
    
    # Sort by creation time
    files_with_times.sort(key=lambda x: x[1])
    
    first_file, first_time = files_with_times[0]
    last_file, last_time = files_with_times[-1]
    
    time_diff = last_time - first_time
    
    # Format output
    print(f"Directory: {directory}")
    print(f"\nFirst created file:")
    print(f"  Path: {first_file}")
    print(f"  Time: {first_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nLast created file:")
    print(f"  Path: {last_file}")
    print(f"  Time: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTime difference: {time_diff}")
    print(f"  Days: {time_diff.days}")
    print(f"  Hours: {time_diff.total_seconds() / 3600:.2f}")
    print(f"  Minutes: {time_diff.total_seconds() / 60:.2f}")

if __name__ == "__main__":
    main()
