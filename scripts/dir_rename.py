import os
import re

def rename_subfolders_with_frame_range(parent_dir):
    print(f"Renaming subfolders in: {parent_dir}")
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        # Skip if folder name already starts with a number
        if re.match(r'^\d', subfolder):
            print(f"Skipping (already starts with number): {subfolder}")
            continue
        # Find all jpg files with numeric names, optionally with _mask
        frame_files = [f for f in os.listdir(subfolder_path) if re.match(r'^(\d+)(_mask)?\.jpg$', f)]
        if not frame_files:
            continue
        frame_numbers = [int(re.match(r'^(\d+)', f).group(1)) for f in frame_files]
        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        new_name = f"{start_frame}_{end_frame}_{subfolder}"
        new_path = os.path.join(parent_dir, new_name)
        os.rename(subfolder_path, new_path)
        print(f"Renamed: {subfolder_path} -> {new_path}")

# if __name__ == "__main__":
    # import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python rename.py <parent_directory>")
    # else:
    #     rename_subfolders_with_frame_range(sys.argv[1])
# parent_dir = r"D:\huayin\1. For Huayin and Alexandra\1. For Huayin and Alexandra\run SAM2 from beginning\AVG 14928 M"
parent_dir=r"D:\huayin\1. For Huayin and Alexandra\1. For Huayin and Alexandra\run SAM2 from beginning\AH2 fix"
rename_subfolders_with_frame_range(parent_dir)