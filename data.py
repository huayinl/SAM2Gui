import os
import re
import shutil
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from tqdm import tqdm
from PIL import Image
import numpy as np
from merge.merge import keep_largest_blob, split_img
import h5py

hdf_path = r"D:\huayin\1. For Huayin and Alexandra\1. For Huayin and Alexandra\run SAM2 from beginning\AH2 fix\recording_10082025_141826_AVG_AH2_crop.hdf5"
mask_dir = r"D:\huayin\1. For Huayin and Alexandra\1. For Huayin and Alexandra\run SAM2 from beginning\AH2 fix\recording_10082025_141826_AVG_AH2_crop.hdf5_masks_20260217_121831_5375_14396"
chosen_side = 'left'  # 'left' or 'right' half of the image to keep from sam2 masks

print(f"Processing video: {hdf_path}")

img_save_dir = r"D:\huayin\SAM2Gui\data\AH2\images"
mask_save_dir = r"D:\huayin\SAM2Gui\data\AH2\masks"
os.makedirs(img_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

# (1) create folder of binary masks from mask_dir
jpg_files = [f for f in os.listdir(mask_dir) if f.endswith('.jpg')]
jpg_files.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))  # sort by frame index
start_frame, end_frame = int(re.search(r'^(\d+)', jpg_files[0]).group(1)), int(re.search(r'^(\d+)', jpg_files[-1]).group(1))
print(f"Found range of frames: {start_frame} to {end_frame} in {mask_dir}")
for filename in tqdm(jpg_files, desc=f"  {mask_dir}", leave=False):
    try:
        # Read JPG and convert to grayscale
        img_pil = Image.open(os.path.join(mask_dir, filename)).convert('L')
        img_gray = np.array(img_pil)

        # Apply binary threshold (> 10) to clean up JPEG artifacts
        binary_mask = (img_gray > 10).astype(np.uint8)
        binary_mask = keep_largest_blob(binary_mask)
        binary_mask = split_img(binary_mask, side=chosen_side)  # take right half only
        
        # Get the frame index from filename (e.g., "2001.jpg" -> "2001")
        frame_index = filename.replace('.jpg', '')
        # Save the binary mask
        save_path = os.path.join(mask_save_dir, f"{frame_index}.png")
        Image.fromarray(binary_mask * 255).save(save_path)
    except Exception as e:
        tqdm.write(f"Error processing {filename}: {e}")

# (2) create folder of png images from hdf5 video for the same range of frames as the masks (assuming the frame index in the mask filename corresponds to the frame index in the video)
with h5py.File(hdf_path, 'r') as f:
    print(f"Keys in HDF5 file: {list(f.keys())}")
    dataset_name = list(f.keys())[0]  # assuming the video data is under the first key
    for frame_index in tqdm(range(int(start_frame), int(end_frame) + 1), desc="Extracting frames from HDF5"):
        try:
            frame_data = f[dataset_name][frame_index]
            frame_data = split_img(frame_data, side=chosen_side)
            if frame_index == 0:
                print(f"Frame shape: {frame_data.shape}, dtype: {frame_data.dtype}")
            img_save_path = os.path.join(img_save_dir, f"{frame_index}.png")
            Image.fromarray(frame_data).save(img_save_path)
        except Exception as e:
            tqdm.write(f"Error processing frame {frame_index}: {e}")


# (3) split into train/val/test sets, ensuring that frames from the same video are in the same set
split_root = r"D:\huayin\SAM2Gui\data\AH2"
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]
all_frames = sorted([f for f in os.listdir(img_save_dir) if f.endswith('.png')])
num_frames = len(all_frames)
train_end = int(num_frames * split_ratios[0])
val_end = train_end + int(num_frames * split_ratios[1])
split_frames = [all_frames[:train_end], all_frames[train_end:val_end], all_frames[val_end:]]

for split, frames in zip(splits, split_frames):
    split_dir = os.path.join(split_root, f"{split}_data_dir")
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    for frame in tqdm(frames, desc=f"Copying {split} frames", leave=False):
        # Copy image
        src_img = os.path.join(img_save_dir, frame)
        dst_img = os.path.join(images_dir, frame)
        shutil.copy(src_img, dst_img)
        # Copy mask, rename to *_mask.png
        src_mask = os.path.join(mask_save_dir, frame)
        frame_base = os.path.splitext(frame)[0]
        dst_mask = os.path.join(masks_dir, f"{frame_base}_mask.png")
        shutil.copy(src_mask, dst_mask)