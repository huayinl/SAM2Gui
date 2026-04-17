import os
sam2_dir = "/Users/huayinluo/Documents/drive-download-20260225T183745Z-1-001/AH2 fix"
output_dir = "/Users/huayinluo/Documents/masks/AH2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# then run merge.py to merge sam2 masks into output_dir, overwriting any duplicates from sam1
import merge
import sys
sys.argv = ['merge.py', sam2_dir, output_dir]
merge.main()
