""" 
Merges the SAM2 masks (in JPG format) into a single folder of PNG masks, with the same frame index as the original video frames. This is done by taking the left half of the SAM2 mask (which corresponds to the original video frame) and saving it as a binary PNG mask. The output PNG files are named with the frame index followed by "_mask.png" (e.g., "2001_mask.png").
"""

import os
import shutil
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import time
import sys

def keep_largest_blob(binary):
    bw = (binary * 255).astype('uint8')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if labels.max() == 0:
        return bw
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_lbl = areas.argmax() + 1
    return (labels == largest_lbl).astype('uint8') * 255

def split_img(img, side):
    # split img in half vertically
    h, w = img.shape
    if side == 'left':
        return img[:, :w//2]
    else:
        return img[:, w//2:]

def main():
    if len(sys.argv) != 3:
        print("Usage: python merge.py <sam2_dir> <output_dir>")
        print("Example: python merge.py /path/to/sam2/masks /path/to/output")
        sys.exit(1)

    start_time = time.time()
    chosen_side = 'left'  # 'left' or 'right' half of the image to keep from sam2 masks
    sam2_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(sam2_dir):
        print(f"Error: sam2_dir '{sam2_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process sam2_dir - convert JPG masks to binary PNG with true frame index
    # Note: this overwrites any duplicates from sam1_dir (since the mask result from SAM2 was a fix over SAM1)
    print("\nProcessing sam2_dir...")
    subfolders_sam2 = [f for f in os.listdir(sam2_dir) if os.path.isdir(os.path.join(sam2_dir, f))]

    for subfolder in tqdm(subfolders_sam2, desc="sam2_dir subfolders"):
        subfolder_path = os.path.join(sam2_dir, subfolder)
        
        # Read all .jpg files in this subfolder
        jpg_files = [f for f in os.listdir(subfolder_path) if f.endswith('.jpg')]
        print(f" Found range of frames: {jpg_files[0]} to {jpg_files[-1]} in {subfolder}")
        for filename in tqdm(jpg_files, desc=f"  {subfolder}", leave=False):
            try:
                # Read JPG and convert to grayscale
                img_pil = Image.open(os.path.join(subfolder_path, filename)).convert('L')
                img_gray = np.array(img_pil)

                # Apply binary threshold (> 10) to clean up JPEG artifacts
                binary_mask = (img_gray > 10).astype(np.uint8)
                binary_mask = keep_largest_blob(binary_mask)
                binary_mask = split_img(binary_mask, side=chosen_side)  # take right half only
                
                # Get the frame index from filename (e.g., "2001.jpg" -> "2001")
                frame_index = filename.replace('.jpg', '')
                
                # Save as PNG to output folder
                output_filename = f"{frame_index}_mask.png"
                output_path = os.path.join(output_dir, output_filename)
                
                Image.fromarray(binary_mask, mode='L').save(output_path)
            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")

    print(f"\nDone! Combined masks saved to {output_dir}")

    print(f"Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()