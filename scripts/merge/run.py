
vid = "RIF_P"

# R23
if vid == "R23":
    sam1_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R23 mask and recording/recording_10292025_121024 _RIF R23 C+M  from SAM1"
    sam2_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R23 mask and recording/R23 fix SAM2 need fixing"
    output_dir="/Users/huayinluo/Documents/masks/R23"

elif vid == "R24":
    # R24
    sam1_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R24 mask and recording/recording_10292025_125639_RIF_R24 C+M"
    sam2_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R24 mask and recording/R24 fix SAM2 done need fixing"
    output_dir="/Users/huayinluo/Documents/masks/R24"

elif vid == "R25":
  # R25
    sam1_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R25 mask and recording/recording_10292025_132934_RIF_R25  C+M"
    sam2_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/R25 mask and recording/R25 fix SAM2 done need fixing"
    output_dir="/Users/huayinluo/Documents/masks/R25"

elif vid == "RIF_P":
    # RIF_P
    sam1_dir = None
    sam2_dir="/Users/huayinluo/Downloads/1. For Huayin and Alexandra/need merging renaming and re-SAM2 to fix/RIF P 15151 mask and recording"
    output_dir="/Users/huayinluo/Documents/masks/RIF_P"


print(f"Processing video: {vid}")
# first run tmp.py to consolidate sam1 masks into output_dir
if sam1_dir:
    import SAM2Gui.merge.consolidate as consolidate
    import sys
    sys.argv = ['tmp.py', sam1_dir, output_dir]
    consolidate.main()


# then run merge.py to merge sam2 masks into output_dir, overwriting any duplicates from sam1

import merge
import sys
sys.argv = ['merge.py', sam2_dir, output_dir]
merge.main()
