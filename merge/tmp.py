import os
import re
import shutil
import sys

def consolidate_frames(parent_dir, dest_folder):
    manifest_file = os.path.join(dest_folder, "processing_manifest.txt")
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Regex to find a range of numbers like "101-1000"
    range_pattern = re.compile(r'(\d+)-(\d+)')
    
    # Store ranges for the manifest
    processed_ranges = []

    # Get list of folders and sort them to process in order (optional but cleaner)
    folders = sorted([f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))])

    for folder_name in folders:
        folder_path = os.path.join(parent_dir, folder_name)
        
        if folder_path == dest_folder:
            continue

        # Extract the start and end frame from the folder name
        match = range_pattern.search(folder_name)
        if not match:
            print(f"Skipping {folder_name}: No frame range found.")
            continue

        start_range = int(match.group(1))
        end_range = int(match.group(2))
        
        print(f"\n--- Folder: {folder_name} ---")
        print(f"Mapping local frames to global range: {start_range} to {end_range}")

        # Track actual frames processed in this folder for the manifest
        files_count = 0
        
        # Loop through files inside the folder
        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.png'):
                continue

            try:
                # Extract local index (e.g., "0" from "0_mask.png")
                local_index = int(re.search(r'^(\d+)', file_name).group(1))
            except (AttributeError, ValueError):
                continue

            # Calculate global frame
            global_frame = start_range + local_index
            new_file_name = f"{global_frame}_mask.png"
            
            src_path = os.path.join(folder_path, file_name)
            dst_path = os.path.join(dest_folder, new_file_name)

            shutil.copy2(src_path, dst_path)
            
            # Detailed logging
            if files_count % 100 == 0: # Log every 100th file to avoid console spam
                print(f"  [LOG] {file_name} -> {new_file_name}")
            
            files_count += 1

        processed_ranges.append({
            "folder": folder_name,
            "range": f"{start_range}-{end_range}",
            "count": files_count
        })

    # Write the manifest file
    with open(manifest_file, "w") as f:
        f.write("FRAME PROCESSING MANIFEST\n")
        f.write("="*30 + "\n")
        for entry in processed_ranges:
            f.write(f"Source Folder: {entry['folder']}\n")
            f.write(f"Frame Range:   {entry['range']}\n")
            f.write(f"Files Moved:   {entry['count']}\n")
            f.write("-" * 30 + "\n")

    print(f"\nDone! All files are in: {dest_folder}")
    print(f"Manifest written to: {manifest_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python tmp.py <parent_directory> <dest_folder>")
        print("Example: python tmp.py /path/to/parent/folder /path/to/output")
        sys.exit(1)
    
    parent_dir = sys.argv[1]
    dest_folder = sys.argv[2]
    
    if not os.path.exists(parent_dir):
        print(f"Error: Directory '{parent_dir}' does not exist.")
        sys.exit(1)
    
    consolidate_frames(parent_dir, dest_folder)

if __name__ == "__main__":
    main()