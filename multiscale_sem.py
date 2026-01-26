import cv2
import numpy as np
from skimage.morphology import skeletonize

# --- for MultiscaleSEM Worm Detection and Centerline Extraction ---
def process_image_and_scale_centers(img, downsample_resolution=8, min_area=1000, max_area=50000, min_hole_area=100000, num_skeleton_points=10):
    """Process image to detect worms and extract skeleton centerline points."""
    original_height, original_width = img.shape[:2]
    new_width = original_width // downsample_resolution
    new_height = original_height // downsample_resolution
    
    downsampled_img = cv2.resize(img, (new_width, new_height))
    
    if len(downsampled_img.shape) == 3:
        gray_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = downsampled_img

    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, initial_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv_mask = 255 - initial_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    filtered_mask = initial_mask.copy()
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            filtered_mask[labels == i] = 255
        elif area > max_area:
            filtered_mask[labels == i] = 255

    filled_mask = filtered_mask.copy()
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    
    for i in range(1, num_labels_white):
        if stats_white[i, cv2.CC_STAT_AREA] < min_hole_area:
            filled_mask[labels_white == i] = 0

    mask = filled_mask
    num_labels_blobs, labels_blobs, stats_blobs, centroids_blobs = cv2.connectedComponentsWithStats(255 - mask, connectivity=8)
    blob_centers = []
    
    for i in range(1, num_labels_blobs):
        area = stats_blobs[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            blob_mask = (labels_blobs == i).astype(np.uint8)
            skeleton = skeletonize(blob_mask)
            skeleton_points = np.where(skeleton)
            
            if len(skeleton_points[0]) > 0:
                all_points = [(skeleton_points[1][j], skeleton_points[0][j]) for j in range(len(skeleton_points[0]))]
                N = len(all_points)
                if N <= num_skeleton_points:
                    blob_skeleton_points = all_points
                else:
                    indices = np.linspace(0, N-1, num_skeleton_points, dtype=int)
                    blob_skeleton_points = [all_points[j] for j in indices]
                blob_centers.append(blob_skeleton_points)

    scaled_blob_centers = []
    for blob_points in blob_centers:
        scaled_points = [(int(cx * downsample_resolution), int(cy * downsample_resolution)) for cx, cy in blob_points]
        scaled_blob_centers.append(scaled_points)
    return scaled_blob_centers

