import os
# vid = "AK2"
# vids = ["R23", "R24", "R25", "RIF_P"]
vids = ["AK2"]
for vid in vids:
    sam2_parent_dir="/Users/huayinluo/Downloads/fixes"
    # R23
    if vid == "R23":
        sam2_dir=os.path.join(sam2_parent_dir, "R23_sam2")
        output_dir="/Users/huayinluo/Documents/masks/R23"

    elif vid == "R24":
        # R24
        sam2_dir=os.path.join(sam2_parent_dir, "R24_sam2")
        output_dir="/Users/huayinluo/Documents/masks/R24"

    elif vid == "R25":
    # R25
        sam2_dir=os.path.join(sam2_parent_dir, "R25_sam2")
        output_dir="/Users/huayinluo/Documents/masks/R25"

    elif vid == "RIF_P":
        # RIF_P
        sam2_dir=os.path.join(sam2_parent_dir, "RIF_sam2")
        output_dir="/Users/huayinluo/Documents/masks/RIF_P"

    elif vid == "AK2":
        # AVG
        sam2_dir=os.path.join(sam2_parent_dir, "AK2_sam2")
        output_dir="/Users/huayinluo/Documents/masks/AVG_AK2"


    # print(f"Processing video: {vid}")
    # # first run tmp.py to consolidate sam1 masks into output_dir
    # if sam1_dir:
    #     import tmp
    #     import sys
    #     sys.argv = ['tmp.py', sam1_dir, output_dir]
    #     tmp.main()


    # then run merge.py to merge sam2 masks into output_dir, overwriting any duplicates from sam1

    import merge
    import sys
    print("--------------------------------")
    print(f"Processing video: {vid}")
    sys.argv = ['merge.py', sam2_dir, output_dir]
    merge.main()
