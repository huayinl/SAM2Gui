""" 
Creates a preview video from a folder of PNG images, compatible with QuickTime on Mac.
"""
import cv2
import os
import re
import sys
from tqdm import tqdm

def create_mac_preview(image_folder, fps=30, scale_factor=0.5):
    if not os.path.exists(image_folder):
        print(f"Error: Folder '{image_folder}' does not exist.")
        return

    folder_name = os.path.basename(os.path.normpath(image_folder))
    # Using .mp4 for QuickTime compatibility
    output_video = os.path.join(image_folder, f"{folder_name}_preview.mp4")

    # 1. Get and sort files numerically
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    try:
        # Handling the {frame}_mask.png naming convention
        images.sort(key=lambda x: int(re.search(r'^(\d+)', x).group(1)))
    except (AttributeError, ValueError):
        print("Error: Could not find numeric frame numbers in filenames.")
        return

    if not images:
        print(f"No PNG files found in {image_folder}")
        return

    # 2. Determine scaled dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape
    new_w, new_h = int(width * scale_factor), int(height * scale_factor)

    # 3. Initialize VideoWriter with mp4v (Native Mac support)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, fps, (new_w, new_h))

    print(f"Creating QuickTime-compatible preview (Scaled to {new_w}x{new_h})...")

    # 4. Write frames with scaling and overlay
    for image in tqdm(images, desc="Encoding MP4", unit="frame"):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        # Resize for low-res preview
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Overlay filename/frame number
        # Scaled font for lower resolution
        cv2.putText(frame, image, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 1, cv2.LINE_AA)
        
        video.write(frame)

    video.release()
    print(f"\nDone! Preview saved to: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/folder [scale_factor]")
    else:
        input_dir = sys.argv[1]
        fps = 60
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        create_mac_preview(input_dir, fps=fps, scale_factor=scale)