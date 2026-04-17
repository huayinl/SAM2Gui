"""
add "_mask" suffix to all png files in the target folder, except those that already have it.
"""
import os
import re

def rename_to_mask(target_folder):
    # Ensure the path exists
    if not os.path.exists(target_folder):
        print(f"Error: Folder '{target_folder}' not found.")
        return

    print(f"Cleaning up files in: {target_folder}")
    
    # Regex to find the leading number/frame count
    frame_pattern = re.compile(r'^(\d+)')
    
    count = 0
    for filename in os.listdir(target_folder):
        # Only look at .png files
        if not filename.lower().endswith('.png'):
            continue
            
        # Skip if it already ends with _mask.png
        if filename.lower().endswith('_mask.png'):
            continue
            
        # Try to extract the frame number
        match = frame_pattern.match(filename)
        if match:
            frame_num = match.group(1)
            new_name = f"{frame_num}_mask.png"
            
            old_path = os.path.join(target_folder, filename)
            new_path = os.path.join(target_folder, new_name)
            
            # Check if destination name already exists to avoid overwriting
            # if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            count += 1
            # else:
            #     print(f"  [SKIP] {filename} -> {new_name} (Destination already exists)")

    print(f"\nRenaming complete. {count} files were updated.")

if __name__ == "__main__":
    # Update this to your 'combined_masks' folder or whichever folder needs fixing
    path_to_clean = "/Users/huayinluo/Documents/masks/AVG"
    rename_to_mask(path_to_clean)