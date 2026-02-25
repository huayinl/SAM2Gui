import os
import re
import sys
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from tqdm import tqdm

def prune_skeleton(binary_mask):
    """Extracts the longest path from a skeletonized binary mask."""
    # 1. Create initial skeleton
    skeleton_img = skeletonize(binary_mask)
    
    # 2. Convert to graph using skan
    # spacing is 1 since we are in pixel space
    try:
        skel_graph = Skeleton(skeleton_img)
        summary = summarize(skel_graph)
        
        # 3. Find the longest branch (path) in the skeleton
        # This effectively 'prunes' all side spurs
        longest_path_idx = summary['euclidean_distance'].idxmax()
        
        # Get the coordinates for the longest branch
        # This returns (y, x) coordinates
        path_coords = skel_graph.path_coordinates(longest_path_idx)
        
        # Flip to (x, y) for consistency
        return path_coords[:, [1, 0]]
        
    except (ValueError, IndexError):
        # Fallback if skeletonization fails or image is empty
        return np.array([])

def process_and_prune_folder(mask_folder, output_file):
    centerline_data = {}
    files = [f for f in os.listdir(mask_folder) if f.endswith('_mask.png')]
    files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))

    print(f"Pruning and extracting centerlines for {len(files)} masks...")

    for filename in tqdm(files):
        mask_path = os.path.join(mask_folder, filename)
        img = Image.open(mask_path).convert('L')
        binary_mask = np.array(img) > 127
        
        # Extract the core path
        core_path = prune_skeleton(binary_mask)
        
        if core_path.size > 0:
            frame_num = int(re.search(r'^(\d+)', filename).group(1))
            centerline_data[frame_num] = core_path

    np.save(output_file, centerline_data)
    print(f"\nDone! Pruned centerlines saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python skeletonize.py <mask_directory>")
        print("Example: python skeletonize.py /Users/huayinluo/Documents/masks/AVG")
        sys.exit(1)
    
    MASK_DIR = sys.argv[1]
    
    if not os.path.exists(MASK_DIR):
        print(f"Error: Directory '{MASK_DIR}' does not exist.")
        sys.exit(1)
    
    OUTPUT_PATH = os.path.join(MASK_DIR, "pruned_centerlines.npy")
    process_and_prune_folder(MASK_DIR, OUTPUT_PATH)