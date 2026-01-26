import sys
import os

# Set MPS memory limit before importing torch
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
import tempfile
import shutil
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import h5py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QStackedWidget, QSlider, QMessageBox, QProgressBar, QComboBox, QScrollArea, QSpinBox, QCheckBox, QMenuBar, QMenu, QAction)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from queue import Queue, Empty
from collections import OrderedDict

from multiscale_sem import process_image_and_scale_centers # for multiscale sem use case
from functools import wraps
from contextlib import contextmanager
import traceback

# --- Check for SAM 2 ---
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 library not found. Tracking will not function.")

# --- utilities ---
def report_errors(func):
    """Decorator to catch errors and emit them via the object's error signal."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            if hasattr(self, 'error'):
                self.error.emit(str(e))
    return wrapper

@contextmanager
def sam2_cleanup(device):
    """Context manager to ensure GPU/MPS memory is cleared after a block of code."""
    try:
        yield
    finally:
        import gc
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

# --- Worker Thread for Heavy SAM 2 Processing ---
class TrackerWorker(QThread):
    finished = pyqtSignal(str) # Emits the path to the mask folder
    progress = pyqtSignal(int) # Optional: to show progress if you added a bar
    error = pyqtSignal(str) # Emits error messages
    maskSaved = pyqtSignal(int, str) # Emits (frame_idx, mask_file_path) when mask is saved
    centerlineComputed = pyqtSignal(int, dict) # Emits (frame_idx, {obj_id: centerline_points})

    def __init__(self, video_path, device, scaled_blob_centers, colors=None, model_size='base', start_frame=0, end_frame=None, video_name=None, num_centerline_points=100, save_mask_dir=None):
        """Initialize TrackerWorker for SAM2 video tracking."""
        super().__init__()
        self.save_mask_dir = save_mask_dir or "/Users/huayinluo/Documents/code/zhenlab/MultiscaleSEM/masks"
        self.video_path = video_path
        self.device = device
        self.scaled_blob_centers = scaled_blob_centers
        self.model_size = model_size
        self.start_frame = int(start_frame) if start_frame is not None else 0
        self.end_frame = int(end_frame) if end_frame is not None else None
        self.video_name = video_name  # Store video_name (h5_file_path or video folder name)
        self.num_centerline_points = int(num_centerline_points) if num_centerline_points is not None else 100
        self._stop_requested = False
        self.chunk_size = 500  # Process 500 frames at a time to avoid OOM
        # Colors can be passed in; if not, fall back to random colors
        if colors is None:
            self.colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        else:
            self.colors = colors # We need colors here now to bake them into the images

    def request_stop(self):
        """Request the worker to stop after the current iteration."""
        self._stop_requested = True

    def _interpolate_centerline(self, centerline, num_points):
        """Select evenly spaced points from centerline."""
        if len(centerline) <= 1:
            return centerline
        
        # Select num_points evenly spaced points
        N = len(centerline)
        if N <= num_points:
            return centerline
        else:
            indices = np.linspace(0, N-1, num_points, dtype=int)
            resampled = [centerline[j] for j in indices]
            return resampled

    def init_sam_predictor(self, model_size):
        """Initialize SAM2 predictor based on model size."""
        if model_size == 'tiny':
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif model_size == 'base':
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif model_size == 'large':
            sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            raise ValueError(f"Unknown model size: {model_size}")

        if not os.path.exists(sam2_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}")

        print(f"Initializing SAM2 predictor with checkpoint: {sam2_checkpoint}")
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        return predictor
    
    def _copy_frame_files(self, src_dir, dst_dir, start_frame, end_frame):
        """Copy frame files from src_dir to dst_dir for frames in [start_frame, end_frame]."""
        frame_files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for absolute_idx in range(start_frame, end_frame + 1):
            src_frame = os.path.join(src_dir, frame_files[absolute_idx])
            dst_frame = os.path.join(dst_dir, f"{absolute_idx - start_frame:06d}.jpg")
            shutil.copy2(src_frame, dst_frame)
    

    def _copy_frame_safe(self, src_frame, dst_frame):
        """Copy a single frame file from src_path to dst_path.
        Try symlink, then hard link, then copy.
        """
        # Try symlink first (fastest, cross-platform)
        try:
            os.symlink(src_frame, dst_frame)
        except (OSError, NotImplementedError):
            # Windows without admin or different filesystem
            # Try hard link (works on same drive, faster than copy)
            try:
                os.link(src_frame, dst_frame)
            except (OSError, NotImplementedError):
                # Different drives or filesystem doesn't support hard links
                # Last resort: copy (slower but always works)
                shutil.copy2(src_frame, dst_frame)
            
    @report_errors
    def run(self):
        """Run tracking in chunks to avoid OOM errors."""
        # Setup output directory for masks
        # --- Create unique timestamped folder ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_output_dir = os.path.join(self.save_mask_dir, f"{self.video_name}_masks_{ts}")
        os.makedirs(mask_output_dir, exist_ok=True)
        print(f"Initialized mask output directory at: {mask_output_dir}")

        # Get total frame count
        frame_files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_frames = len(frame_files)
        actual_end_frame = self.end_frame if self.end_frame is not None else total_frames - 1
        
        # Initialize SAM2 predictor ONCE (will be reused for all chunks)
        model_size = getattr(self, 'model_size', 'base')
        predictor = self.init_sam_predictor(model_size)
        
        # Initial prompts for first chunk
        current_prompts = self.scaled_blob_centers
        # Initialize centerlines storage
        all_centerlines = {}
        
        # Process in chunks
        chunk_start = self.start_frame
        while chunk_start <= actual_end_frame and not self._stop_requested:
            chunk_end = min(chunk_start + self.chunk_size - 1, actual_end_frame)
            print(f"\n=== Tracking chunk: frames {chunk_start} to {chunk_end} ===")
                
            # (1) make a temp dir for this chunk
            # since SAM2 requires a video folder input
            tracking_temp_dir = tempfile.mkdtemp(prefix="sam2_tracking_chunk_")
            for relative_idx, absolute_idx in enumerate(range(chunk_start, chunk_end + 1)):
                src_frame = os.path.join(self.video_path, frame_files[absolute_idx])
                dst_frame = os.path.join(tracking_temp_dir, f"{relative_idx:06d}.jpg")
                self._copy_frame_safe(src_frame, dst_frame)
            print(f"Created tracking subset folder {tracking_temp_dir} with frames {chunk_start} to {chunk_end}")

            # Use Context Manager for automatic cleanup of temp files and GPU memory
            with sam2_cleanup(self.device):
                try:
                    inference_state = predictor.init_state(video_path=tracking_temp_dir)
                    chunk_results = self._track_chunk(predictor, inference_state, chunk_start, chunk_end, current_prompts, mask_output_dir)
                    all_centerlines.update(chunk_results)
                    
                    if chunk_end < actual_end_frame:
                        current_prompts = self._centerlines_to_prompts(chunk_results.get(chunk_end, {}))
                        if not any(current_prompts): break
                finally:
                    shutil.rmtree(tracking_temp_dir)
            # Move to next chunk
            chunk_start = chunk_end + 1
        
        # Save all centerlines
        centerlines_path = os.path.join(mask_output_dir, "centerlines.npy")
        try:
            np.save(centerlines_path, all_centerlines)
            print(f"Saved centerlines to {centerlines_path}")
        except Exception as e:
            print(f"Failed to save centerlines: {e}")
        
        self.finished.emit(mask_output_dir)


    def _track_chunk(self, predictor, inference_state, chunk_start, chunk_end, prompts, mask_output_dir):
        """Track a single chunk of frames using pre-initialized predictor and inference_state."""
        chunk_centerlines = {}
        
        # Add prompts at chunk_start frame
        for i, obj_prompts in enumerate(prompts):
            if not obj_prompts:
                continue
            ann_obj_id = i + 1
            points = np.array([pt for pt, _ in obj_prompts], dtype=np.float32)
            labels = np.array([lbl for _, lbl in obj_prompts], dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        # Get reference image dimensions
        frame_files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        first_frame_path = os.path.join(self.video_path, frame_files[chunk_start])
        ref_img = cv2.imread(first_frame_path)
        h, w = ref_img.shape[:2]

        # Propagate with memory management
        with torch.inference_mode():
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if self._stop_requested:
                    break
                
                save_frame_index = chunk_start + out_frame_idx
                
                # Stop if beyond chunk end
                if save_frame_index > chunk_end:
                    break

                # Create color mask
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                frame_centerlines = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    
                    if mask.sum() > 0:
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        color = self.colors[out_obj_id % len(self.colors)]
                        color_mask[mask > 0] = color
                        
                        # Extract centerline
                        try:
                            skeleton = skeletonize(mask.astype(bool))
                            skeleton_coords = np.column_stack(np.where(skeleton))
                            if len(skeleton_coords) > 0:
                                full_centerline = [(int(pt[1]), int(pt[0])) for pt in skeleton_coords]
                                resampled_centerline = full_centerline
                                if len(full_centerline) > 1:
                                    resampled_centerline = self._interpolate_centerline(full_centerline, self.num_centerline_points)
                                frame_centerlines[out_obj_id] = (full_centerline, resampled_centerline)
                        except Exception as e:
                            print(f"Failed to compute centerline for object {out_obj_id} in frame {save_frame_index}: {e}")
                            frame_centerlines[out_obj_id] = ([], [])

                # Save mask
                save_path = os.path.join(mask_output_dir, f"{save_frame_index}.png")
                cv2.imwrite(save_path, color_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                
                # Store centerlines
                chunk_centerlines[save_frame_index] = frame_centerlines
                
                # Emit signals
                self.maskSaved.emit(save_frame_index, save_path)
                self.centerlineComputed.emit(save_frame_index, frame_centerlines)
                
                # Clean up tensors
                try:
                    del out_mask_logits
                except Exception:
                    pass
                
                # Clear cache periodically
                if save_frame_index % 50 == 0:
                    if self.device.type == "mps":
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    elif self.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

        # Don't cleanup predictor/inference_state here - they're reused across chunks
        # Only do lightweight cache clearing
        import gc
        gc.collect()
        if self.device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        elif self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        return chunk_centerlines

    def _centerlines_to_prompts(self, centerline_dict):
        """Convert centerlines from last frame to prompts for next chunk."""
        prompts = []
        
        # Get all object IDs (sorted to maintain order)
        obj_ids = sorted(centerline_dict.keys())
        
        for obj_id in obj_ids:
            centerline_data = centerline_dict.get(obj_id, ([], []))
            if isinstance(centerline_data, tuple) and len(centerline_data) == 2:
                _, resampled_points = centerline_data
            else:
                resampled_points = centerline_data if centerline_data else []
            
            # Convert resampled centerline points to positive prompts
            obj_prompts = [((x, y), 1) for x, y in resampled_points]
            prompts.append(obj_prompts)
        
        return prompts

# --- Background Loading Components ---

class ImageLoader(QThread):
    """Background loader that reads frames and (optionally) pre-colored masks,
    composes them (mask blending) and emits RGB numpy arrays to the GUI thread.
    The GUI thread converts to QPixmap and caches the scaled pixmap.
    """
    frameLoaded = pyqtSignal(int, object)

    def __init__(self, parent=None):
        """Initialize ImageLoader background thread for loading frames."""
        super().__init__(parent)
        self._req_q = Queue()
        self._running = True
        # references (set by GUI) to lists/dicts owned by GUI
        self.image_files = None
        self.mask_files = None
        # optional HDF5 reference
        self.h5_path = None
        self.h5_dataset = None
        # internal HDF5 handle opened in this thread (open once for efficiency)
        self._h5f = None
        self._h5_n_frames = None

    def set_references(self, image_files=None, mask_files=None, h5_path=None, h5_dataset=None, gui_instance=None):
        """Set references to data sources (image files, masks, or HDF5) for loading frames."""
        self.image_files = image_files
        self.mask_files = mask_files
        self.h5_path = h5_path
        self.h5_dataset = h5_dataset
        self.gui_instance = gui_instance  # Reference to GUI for opacity value
        # reset per-thread HDF5 info; actual file handle will be opened inside run()
        self._h5_n_frames = None
        # if image_files given and it's a list, we keep it; otherwise None

    def request(self, idx):
        """Request loading of a specific frame by index."""
        try:
            self._req_q.put_nowait(idx)
        except Exception:
            try:
                self._req_q.put(idx)
            except Exception:
                pass

    def run(self):
        """Main loop for loading and composing frames in background thread."""
        while self._running:
            try:
                idx = self._req_q.get(timeout=0.1)
            except Empty:
                continue
            if idx is None:
                continue
            try:
                # Allow either file-based image list or HDF5-backed dataset
                if self.image_files is None and self.h5_path is None:
                    # nothing to load
                    continue

                # If using HDF5, ensure we have dataset length and an open handle
                if self.h5_path is not None:
                    # open HDF5 file once in this thread for efficiency
                    if self._h5f is None:
                        try:
                            self._h5f = h5py.File(self.h5_path, 'r')
                            ds_name = self.h5_dataset
                            if ds_name is None:
                                # pick the first dataset
                                ds_name = next(iter(self._h5f.keys()))
                            self._h5_dataset_name_internal = ds_name
                            self._h5_n_frames = self._h5f[ds_name].shape[0]
                        except Exception:
                            # failed to open HDF5 in loader thread
                            try:
                                if self._h5f is not None:
                                    self._h5f.close()
                                self._h5f = None
                            except Exception:
                                pass
                            continue

                    # bounds check using HDF5 length
                    if idx < 0 or (self._h5_n_frames is not None and idx >= self._h5_n_frames):
                        continue

                    try:
                        ds = self._h5f[self._h5_dataset_name_internal]
                        arr = np.asarray(ds[idx])
                        # if grayscale expand to 3 channels
                        if arr.ndim == 2:
                            img_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                        elif arr.ndim == 3 and arr.shape[2] == 3:
                            if arr.dtype != np.uint8:
                                # normalize to 0-255
                                arrf = arr.astype(np.float32)
                                m = arrf.max()
                                if m == 0:
                                    img_rgb = (arrf * 0).astype(np.uint8)
                                else:
                                    img_rgb = (255.0 * (arrf / m)).astype(np.uint8)
                            else:
                                img_rgb = arr.astype(np.uint8)
                        else:
                            # unsupported shape
                            continue
                    except Exception:
                        continue
                else:
                    # file-based images
                    if idx < 0 or idx >= len(self.image_files):
                        continue
                    path = self.image_files[idx]
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # compose mask if available
                if self.mask_files and idx in self.mask_files:
                    mask_path = self.mask_files[idx]
                    mask_bgr = cv2.imread(mask_path)
                    if mask_bgr is not None:
                        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
                        if mask_rgb.shape[:2] != img_rgb.shape[:2]:
                            mask_rgb = cv2.resize(mask_rgb, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_indices = np.any(mask_rgb != [0, 0, 0], axis=-1)
                        # Get opacity from GUI (default 0.5 if not available)
                        opacity = 0.5
                        if hasattr(self, 'gui_instance') and self.gui_instance is not None:
                            try:
                                opacity = self.gui_instance.mask_opacity
                            except Exception:
                                pass
                        # ensure copy before modifying
                        img_rgb = img_rgb.copy()
                        img_rgb[mask_indices] = cv2.addWeighted(img_rgb[mask_indices], 1.0 - opacity, mask_rgb[mask_indices], opacity, 0)

                # emit the composed RGB image (numpy array)
                self.frameLoaded.emit(idx, img_rgb)
            except Exception:
                # ignore per-frame errors
                continue

    def stop(self):
        """Stop the image loader thread."""
        self._running = False
        try:
            self._req_q.put(None)
        except Exception:
            pass

class ExportH5Worker(QThread):
    """Export frames from an HDF5 dataset to a temporary folder as JPEGs.
    Emits `finished(temp_folder)` when done, `progress(int)` updates, and `error(str)` on failure.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, h5_path, start_frame, end_frame, dataset_name=None):
        """Initialize ExportH5Worker to export HDF5 frames to temporary folder."""
        super().__init__()
        self.h5_path = h5_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.dataset_name = dataset_name
        # stream_mode: if True, emit `ready` after writing initial frames so consumer can start
        self.stream_mode = False
        self.stream_threshold = 10
        print(f"Initialized ExportH5Worker with start_frame={start_frame}, end_frame={end_frame}, dataset_name={dataset_name}")

    def run(self):
        """Export HDF5 frames to temporary folder as JPEG files."""
        try:
            with h5py.File(self.h5_path, 'r') as f:
                ds_name = self.dataset_name
                if ds_name is None:
                    # pick preferred dataset names or first dataset
                    for pref in ('images', 'frames', 'data'):
                        if pref in f:
                            ds_name = pref
                            break
                if ds_name is None:
                    for k in f.keys():
                        if isinstance(f[k], h5py.Dataset):
                            ds_name = k
                            break
                if ds_name is None:
                    self.error.emit('No dataset found in HDF5')
                    return

                ds = f[ds_name]
                # n_frames = ds.shape[0]
                n_frames = self.end_frame - self.start_frame + 1
                temp_dir = tempfile.mkdtemp(prefix='h5_frames_')
                print(f"{self.start_frame}, {self.end_frame}")
                for i in tqdm(range(self.start_frame, self.end_frame + 1), desc=f"Exporting frames from HDF5..."):
                    try:
                        arr = np.asarray(ds[i])
                        if arr.ndim == 2:
                            out = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        elif arr.ndim == 3 and arr.shape[2] == 3:
                            if arr.dtype != np.uint8:
                                # normalize to 0-255
                                arrf = arr.astype(np.float32)
                                m = arrf.max()
                                if m == 0:
                                    out = (arrf * 0).astype(np.uint8)
                                else:
                                    out = (255.0 * (arrf / m)).astype(np.uint8)
                            else:
                                out = arr.astype(np.uint8)
                            # convert RGB->BGR if needed (assume RGB)
                            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                        else:
                            # unsupported dim
                            continue

                        save_path = os.path.join(temp_dir, f"{i}.jpg")
                        cv2.imwrite(save_path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        frames_processed = i - self.start_frame + 1
                        if frames_processed % max(1, n_frames // 100) == 0:
                            pct = int(100.0 * frames_processed / n_frames)
                            self.progress.emit(pct)

                        # if streaming, emit ready once we've written enough frames to start
                        if self.stream_mode and frames_processed == min(self.stream_threshold, n_frames):
                            try:
                                print("Emitting ready signal from ExportH5Worker")
                                self.ready.emit(temp_dir)
                            except Exception:
                                pass
                        elif frames_processed == n_frames:
                            try:
                                print("Emitting ready signal from ExportH5Worker")
                                self.ready.emit(temp_dir)
                            except Exception:
                                pass                          
                    except Exception as e:
                        print(f"Skipping processing frame {i}: {e}")
                        # skip problematic frames
                        continue
            self.finished.emit(temp_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

# --- Main GUI Application ---

class C_Elegans_GUI(QMainWindow):
    def __init__(self):
        """Initialize the main GUI window and setup all UI components."""
        super().__init__()
        self.setWindowTitle("C. Elegans Tracker (SAM 2)")
        self.resize(1000, 800)

        # State Variables
        self.video_path = None
        self.image_files = [] # sorted list of full paths
        self.current_frame_idx = 0
        # store last loaded RGB full-resolution frame for coordinate mapping
        self._last_loaded_rgb = None
        # hovered prompt point state: (obj_idx, point_idx) or None
        self._hovered_point = None
        # Prompts storage: object-centric mapping
        # obj_id -> {frame_idx -> [((x,y), label), ((x,y), label), ...], ...}
        # where label is 1 (positive) or 0 (negative)
        self.prompts_by_object = {}
        # Keep track of next object ID to assign
        self.next_object_id = 0
        self.autoseg_frame_idx = None
        self.tracking_results = None # Stores result of SAM 2
        self.colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8) # Pre-generate 100 random colors
        # Mapping from frame_idx -> mask file path (pre-colored JPGs)
        self.tracking_mask_files = {}
        # Mapping from frame_idx -> {obj_id: [(x, y), ...]} centerline points
        self.centerlines = {}
        # Mask opacity (0.0 to 1.0, where 1.0 = fully opaque)
        self.mask_opacity = 0.5
        # Prompt point size (radius for circles, half-size for X markers)
        self.prompt_point_size = 15
        # Simple in-memory cache for recently-used color masks to speed up display
        self._mask_cache = {}
        self._mask_cache_max = 64
        # Pixmap cache (scaled QPixmaps) for fast display
        # Use OrderedDict for LRU behavior (most-recently-used at end)
        self.pixmap_cache = OrderedDict()
        self.pixmap_cache_max = 512
        # Prefetch radius (number of frames ahead/behind to load)
        self.prefetch_radius = 3
        # Playback state
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(100)  # default 100ms -> 10 FPS; adjust as needed
        self.play_timer.timeout.connect(self._on_play_timeout)
        self.playing = False

        # Ensure main window can receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Setup Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # UI Setup
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Setup Menu Bar
        self.setup_menu_bar()

        # Mask output directory (initialize before setup_tracking_screen)
        self.save_mask_dir = "/Users/huayinluo/Documents/code/zhenlab/MultiscaleSEM/masks"

        self.setup_selection_screen()
        self.setup_tracking_screen()

        # Background image loader thread
        self._image_loader = ImageLoader(self)
        self._image_loader.frameLoaded.connect(self._on_frame_loaded)
        self._image_loader.start()
        # initial references (empty) - will be updated when folder selected / tracking finished
        self._image_loader.set_references(self.image_files, self.tracking_mask_files, gui_instance=self)
        # temp dir used when exporting HDF5 frames for tracking
        self._h5_export_tempdir = None
        # track HDF5 file path and dataset name
        self.h5_file_path = None
        self.h5_dataset_name = None
        # track TIF temp directory
        self.tif_temp_dir = None

    # --- Menu Bar Setup ---
    
    def setup_menu_bar(self):
        """Setup the menu bar with File menu and import options."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Import New Folder action
        import_folder_action = QAction('Import New Folder', self)
        import_folder_action.triggered.connect(self.import_new_folder)
        file_menu.addAction(import_folder_action)
        
        # Import New TIFF action
        import_tiff_action = QAction('Import New TIFF', self)
        import_tiff_action.triggered.connect(self.import_new_tiff)
        file_menu.addAction(import_tiff_action)
        
        # Import New HDF5 action
        import_hdf5_action = QAction('Import New HDF5', self)
        import_hdf5_action.triggered.connect(self.import_new_hdf5)
        file_menu.addAction(import_hdf5_action)

    # --- State Management ---
    
    def reset_state(self):
        """Reset all state variables and clean up memory when loading a new video."""
        # Stop playback if active
        if self.playing:
            self.playing = False
            self.play_timer.stop()
            if hasattr(self, 'btn_play'):
                self.btn_play.setText("▶ Play")
        
        # Stop any ongoing tracking
        if hasattr(self, 'tracker_worker') and self.tracker_worker is not None:
            try:
                self.tracker_worker.request_stop()
                self.tracker_worker.wait(2000)  # Wait up to 2 seconds
                self.tracker_worker = None
            except Exception:
                pass
        
        # Clean up temporary directories
        if hasattr(self, 'tif_temp_dir') and self.tif_temp_dir and os.path.exists(self.tif_temp_dir):
            try:
                shutil.rmtree(self.tif_temp_dir)
                self.tif_temp_dir = None
            except Exception as e:
                print(f"Warning: Could not remove temp dir {self.tif_temp_dir}: {e}")
        
        if hasattr(self, '_h5_export_tempdir') and self._h5_export_tempdir and os.path.exists(self._h5_export_tempdir):
            try:
                shutil.rmtree(self._h5_export_tempdir)
                self._h5_export_tempdir = None
            except Exception as e:
                print(f"Warning: Could not remove H5 export temp dir: {e}")
        
        # Reset all state variables
        self.video_path = None
        self.image_files = []
        self.current_frame_idx = 0
        self._last_loaded_rgb = None
        self._hovered_point = None
        self.prompts_by_object = {}
        self.next_object_id = 0
        self.autoseg_frame_idx = None
        self.tracking_results = None
        self.tracking_mask_files = {}
        self.centerlines = {}
        self.h5_file_path = None
        self.h5_dataset_name = None
        
        # Clear all caches
        self._mask_cache = {}
        self.pixmap_cache = OrderedDict()
        
        # Reset UI elements
        if hasattr(self, 'slider'):
            self.slider.setValue(0)
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
        
        if hasattr(self, 'frame_spinbox'):
            self.frame_spinbox.setValue(0)
            self.frame_spinbox.setMaximum(0)
            self.frame_spinbox.setEnabled(False)
        
        if hasattr(self, 'btn_play'):
            self.btn_play.setEnabled(False)
        
        if hasattr(self, 'btn_track'):
            self.btn_track.setEnabled(True)
        
        if hasattr(self, 'status_label'):
            self.status_label.setText("Ready")
        
        # Update image loader to clear references
        try:
            self._image_loader.set_references([], {}, gui_instance=self)
        except Exception:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def import_new_folder(self):
        """Import a new video folder after cleaning up the current state."""
        self.reset_state()
        self.choose_folder()

    def import_new_tiff(self):
        """Import a new TIFF file after cleaning up the current state."""
        self.reset_state()
        self.choose_tif()

    def import_new_hdf5(self):
        """Import a new HDF5 file after cleaning up the current state."""
        self.reset_state()
        self.choose_hdf5()

    # --- Screen Setup ---
    
    def setup_selection_screen(self):
        """Setup the initial selection screen for choosing video source."""
        self.selection_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("Please import a video folder or HDF5/TIF file to begin")
        label.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        
        btn_select = QPushButton("Select Video Folder")
        btn_select.setFixedSize(200, 50)
        btn_select.clicked.connect(self.choose_folder)

        btn_import_h5 = QPushButton("Import HDF5 File")
        btn_import_h5.setFixedSize(200, 50)
        btn_import_h5.clicked.connect(self.choose_hdf5)

        btn_import_tif = QPushButton("Import TIF Video")
        btn_import_tif.setFixedSize(200, 50)
        btn_import_tif.clicked.connect(self.choose_tif)

        layout.addWidget(label)
        layout.addWidget(btn_select, alignment=Qt.AlignCenter)
        layout.addWidget(btn_import_h5, alignment=Qt.AlignCenter)
        layout.addWidget(btn_import_tif, alignment=Qt.AlignCenter)
        self.selection_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.selection_widget)

    # --- File and Data Loading ---
    
    def choose_hdf5(self):
        """Open file dialog to select and load an HDF5 file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select HDF5 file", filter="HDF5 Files (*.h5 *.hdf5);;All Files (*)")
        if not path:
            return

        # Attempt to inspect the file to find a suitable dataset
        try:
            with h5py.File(path, 'r') as f:
                # prefer dataset named 'images' or 'frames', otherwise take first dataset
                ds_name = None
                for pref in ('images', 'frames', 'data'):
                    if pref in f:
                        ds_name = pref
                        break
                # pick first dataset
                if ds_name is None:
                    for k in f.keys():
                        if isinstance(f[k], h5py.Dataset):
                            ds_name = k
                            break
                if ds_name is None:
                    QMessageBox.critical(self, "Error", "No dataset found in HDF5 file.")
                    return
                n_frames = f[ds_name].shape[0]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read HDF5 file: {e}")
            return

        # set up HDF5-backed dataset
        self.video_path = None
        self.image_files = None
        self.h5_file_path = path
        self.h5_dataset_name = ds_name

        # configure slider and UI
        self.slider.setMaximum(max(0, n_frames - 1))
        self.slider.setEnabled(True)
        self.frame_spinbox.setMaximum(max(0, n_frames - 1))
        self.frame_spinbox.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.stacked_widget.setCurrentIndex(1)

        # update image loader references and clear caches
        try:
            self._image_loader.set_references(image_files=None, mask_files=self.tracking_mask_files, h5_path=self.h5_file_path, h5_dataset=self.h5_dataset_name, gui_instance=self)
        except Exception:
            pass
        self.clear_pixmap_cache()
        self.update_display()
        # disable tracking (TrackerWorker expects a folder of frames)
        try:
            self.btn_track.setEnabled(False)
        except Exception:
            pass
        self.status_label.setText(f"Loaded HDF5: {os.path.basename(path)} ({n_frames} frames)")
        # Prefetch first frames
        try:
            self._image_loader.request(self.current_frame_idx)
            self._prefetch_neighbors(self.current_frame_idx)
        except Exception:
            pass
        # update left-sidebar spinboxes if present
        try:
            if hasattr(self, 'end_spin'):
                self.end_spin.setMaximum(max(0, n_frames - 1))
                self.end_spin.setValue(max(0, n_frames - 1))
            if hasattr(self, 'start_spin'):
                self.start_spin.setMaximum(max(0, n_frames - 1))
                self.start_spin.setValue(0)
        except Exception:
            pass

    def choose_tif(self):
        """Import a TIF/TIFF video file (multi-frame TIFF stack)."""
        path, _ = QFileDialog.getOpenFileName(self, "Select TIF/TIFF file", filter="TIF Files (*.tif *.tiff);;All Files (*)")
        if not path:
            return

        # Open the TIFF file to check how many frames it has
        try:
            with Image.open(path) as img:
                n_frames = getattr(img, 'n_frames', 1)
            
            if n_frames is None:
                n_frames = 1
            
            self.status_label.setText(f"Loading TIF: {os.path.basename(path)} ({n_frames} frames)...")
            
            # Create a temporary directory to extract frames
            temp_dir = tempfile.mkdtemp(prefix="tif_video_")
            self.tif_temp_dir = temp_dir  # Track temp dir for cleanup
            
            # Extract frames from TIFF as JPG (required for SAM2 tracker)
            with Image.open(path) as img:
                for frame_idx in range(n_frames):
                    try:
                        img.seek(frame_idx)
                        frame = img.convert('RGB')
                        frame_path = os.path.join(temp_dir, f"{frame_idx:06d}.jpg")
                        frame.save(frame_path, quality=95)
                    except EOFError:
                        # End of frames
                        break
            
            # Load extracted frames just like a folder
            self.video_path = temp_dir
            files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            
            self.image_files = [os.path.join(temp_dir, f) for f in files]
            
            if not self.image_files:
                QMessageBox.warning(self, "Error", "No frames could be extracted from the TIF file.")
                shutil.rmtree(temp_dir)
                return
            
            # Configure UI
            self.slider.setMaximum(len(self.image_files) - 1)
            self.slider.setEnabled(True)
            self.frame_spinbox.setMaximum(len(self.image_files) - 1)
            self.frame_spinbox.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.stacked_widget.setCurrentIndex(1)
            
            # Update image loader references
            try:
                self._image_loader.set_references(self.image_files, self.tracking_mask_files, gui_instance=self)
            except Exception:
                pass
            
            self.clear_pixmap_cache()
            self.update_display()
            self.status_label.setText(f"Loaded TIF: {os.path.basename(path)} ({len(self.image_files)} frames)")
            
            # Prefetch first frames
            try:
                self._image_loader.request(self.current_frame_idx)
                self._prefetch_neighbors(self.current_frame_idx)
            except Exception:
                pass
            
            # Update left-sidebar spinboxes if present
            try:
                n_frames = len(self.image_files)
                if hasattr(self, 'end_spin'):
                    self.end_spin.setMaximum(max(0, n_frames - 1))
                    self.end_spin.setValue(max(0, n_frames - 1))
                if hasattr(self, 'start_spin'):
                    self.start_spin.setMaximum(max(0, n_frames - 1))
                    self.start_spin.setValue(0)
            except Exception:
                pass
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIF file: {str(e)}")
            return

    # --- Object and Prompt Management ---

    def get_all_object_ids(self):
        """Get sorted list of all object IDs that exist across all frames."""
        return sorted(self.prompts_by_object.keys())

    def get_prompts_for_frame(self, frame_idx):
        """Get prompts for all objects in a frame. Returns list indexed by object_id.
        For each object: [(x,y), (x,y), ...] or empty list if no prompts for this frame."""
        all_objects = self.get_all_object_ids()
        result = []
        for obj_id in all_objects:
            if obj_id in self.prompts_by_object:
                frame_prompts = self.prompts_by_object[obj_id].get(frame_idx, [])
                result.append(frame_prompts)
            else:
                result.append([])
        return result

    def get_prompts_for_object_in_frame(self, obj_id, frame_idx):
        """Get prompts for a specific object in a specific frame."""
        if obj_id not in self.prompts_by_object:
            return []
        return self.prompts_by_object[obj_id].get(frame_idx, [])

    def set_prompts_for_object_in_frame(self, obj_id, frame_idx, prompts):
        """Set prompts for a specific object in a specific frame.
        prompts: list of ((x, y), label) tuples where label is 1 (positive) or 0 (negative)
        """
        if obj_id not in self.prompts_by_object:
            self.prompts_by_object[obj_id] = {}
        self.prompts_by_object[obj_id][frame_idx] = prompts

    def create_new_object(self, frame_idx, initial_prompts=None):
        """Create a new object and optionally add initial prompts in a frame.
        Returns the new object_id."""
        obj_id = self.next_object_id
        self.next_object_id += 1
        self.prompts_by_object[obj_id] = {}
        if initial_prompts is not None:
            self.prompts_by_object[obj_id][frame_idx] = initial_prompts
        return obj_id

    def setup_tracking_screen(self):
        """Setup the main tracking screen with image display and controls."""
        self.tracking_widget = QWidget()
        layout = QVBoxLayout()

        # Image Display Area
        # Left: Image display
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: #222;")

        # Right: Sidebar for object buttons
        self.sidebar_area = QScrollArea()
        self.sidebar_area.setWidgetResizable(True)
        self.sidebar_content = QWidget()
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_content.setLayout(self.sidebar_layout)
        self.sidebar_area.setWidget(self.sidebar_content)
        self.sidebar_area.setFixedWidth(240)

        # Container for object buttons (will be populated after autoseg)
        self.object_button_container = QWidget()
        self.object_button_layout = QVBoxLayout()
        self.object_button_layout.setAlignment(Qt.AlignTop)
        self.object_button_container.setLayout(self.object_button_layout)
        # put a title label at top
        self.sidebar_layout.addWidget(QLabel("Objects:"))
        self.sidebar_layout.addWidget(self.object_button_container)

        # enable mouse events on the image label and install event filter for clicks
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)

        # Place image and sidebar side-by-side
        # Left: control sidebar (track buttons + start/end frame)
        self.left_sidebar = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_layout.setAlignment(Qt.AlignTop)
        self.left_sidebar.setLayout(self.left_layout)

        # Mask output directory row
        mask_dir_widget = QWidget()
        mask_dir_layout = QVBoxLayout()
        mask_dir_layout.setContentsMargins(0, 0, 0, 0)
        mask_dir_widget.setLayout(mask_dir_layout)
        mask_dir_layout.addWidget(QLabel("Mask Output Dir:"))
        self.mask_dir_label = QLabel(self.save_mask_dir)
        self.mask_dir_label.setWordWrap(True)
        self.mask_dir_label.setStyleSheet("font-size: 9px; color: #888;")
        mask_dir_layout.addWidget(self.mask_dir_label)
        btn_choose_mask_dir = QPushButton("Browse...")
        btn_choose_mask_dir.setFixedHeight(28)
        btn_choose_mask_dir.clicked.connect(self.choose_mask_output_dir)
        mask_dir_layout.addWidget(btn_choose_mask_dir)
        self.left_layout.addWidget(mask_dir_widget)
        self.left_layout.addSpacing(10)

        self.left_layout.addWidget(QLabel("Tracking Range:"))
        
        # Start frame row
        start_row_widget = QWidget()
        start_row_layout = QHBoxLayout()
        start_row_layout.setContentsMargins(0, 0, 0, 0)
        start_row_widget.setLayout(start_row_layout)
        start_row_layout.addWidget(QLabel("Start:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(0)
        self.start_spin.setMaximum(0)
        self.start_spin.setValue(0)
        start_row_layout.addWidget(self.start_spin)
        btn_set_start = QPushButton("Set to Curr")
        btn_set_start.setFixedWidth(80)
        btn_set_start.clicked.connect(lambda: self.start_spin.setValue(self.current_frame_idx))
        start_row_layout.addWidget(btn_set_start)
        start_row_layout.addStretch()
        self.left_layout.addWidget(start_row_widget)
        
        # End frame row
        end_row_widget = QWidget()
        end_row_layout = QHBoxLayout()
        end_row_layout.setContentsMargins(0, 0, 0, 0)
        end_row_widget.setLayout(end_row_layout)
        end_row_layout.addWidget(QLabel("End:"))
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(0)
        self.end_spin.setMaximum(0)
        self.end_spin.setValue(0)
        end_row_layout.addWidget(self.end_spin)
        btn_set_end = QPushButton("Set to Curr")
        btn_set_end.setFixedWidth(80)
        btn_set_end.clicked.connect(lambda: self.end_spin.setValue(self.current_frame_idx))
        end_row_layout.addWidget(btn_set_end)
        end_row_layout.addStretch()
        self.left_layout.addWidget(end_row_widget)

        # Stream mode row
        stream_row_widget = QWidget()
        stream_row_layout = QHBoxLayout()
        stream_row_layout.setContentsMargins(0, 0, 0, 0)
        stream_row_widget.setLayout(stream_row_layout)
        self.stream_mode_checkbox = QCheckBox("Stream Mode")
        self.stream_mode_checkbox.setChecked(False)
        stream_row_layout.addWidget(self.stream_mode_checkbox)
        stream_row_layout.addWidget(QLabel("Threshold:"))
        self.stream_threshold_spin = QSpinBox()
        self.stream_threshold_spin.setMinimum(1)
        self.stream_threshold_spin.setMaximum(1000)
        self.stream_threshold_spin.setValue(10)
        self.stream_threshold_spin.setFixedWidth(60)
        stream_row_layout.addWidget(self.stream_threshold_spin)
        stream_row_layout.addStretch()
        self.left_layout.addWidget(stream_row_widget)

        # Show centerlines checkbox
        centerline_checkbox_widget = QWidget()
        centerline_checkbox_layout = QHBoxLayout()
        centerline_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        centerline_checkbox_widget.setLayout(centerline_checkbox_layout)
        self.show_centerlines_checkbox = QCheckBox("Show Centerlines")
        self.show_centerlines_checkbox.setChecked(True)
        self.show_centerlines_checkbox.stateChanged.connect(self._on_show_centerlines_changed)
        centerline_checkbox_layout.addWidget(self.show_centerlines_checkbox)
        centerline_checkbox_layout.addStretch()
        self.left_layout.addWidget(centerline_checkbox_widget)

        # Centerline points row
        centerline_row_widget = QWidget()
        centerline_row_layout = QHBoxLayout()
        centerline_row_layout.setContentsMargins(0, 0, 0, 0)
        centerline_row_widget.setLayout(centerline_row_layout)
        centerline_row_layout.addWidget(QLabel("Centerline Pts:"))
        self.centerline_points_spin = QSpinBox()
        self.centerline_points_spin.setMinimum(1)
        self.centerline_points_spin.setMaximum(1000)
        self.centerline_points_spin.setValue(100)
        self.centerline_points_spin.setFixedWidth(60)
        self.centerline_points_spin.valueChanged.connect(self._on_centerline_points_changed)
        centerline_row_layout.addWidget(self.centerline_points_spin)
        centerline_row_layout.addStretch()
        self.left_layout.addWidget(centerline_row_widget)

        # Prompt point size row
        prompt_point_row_widget = QWidget()
        prompt_point_row_layout = QHBoxLayout()
        prompt_point_row_layout.setContentsMargins(0, 0, 0, 0)
        prompt_point_row_widget.setLayout(prompt_point_row_layout)
        prompt_point_row_layout.addWidget(QLabel("Point Size:"))
        self.prompt_point_size_spin = QSpinBox()
        self.prompt_point_size_spin.setMinimum(1)
        self.prompt_point_size_spin.setMaximum(50)
        self.prompt_point_size_spin.setValue(15)
        self.prompt_point_size_spin.setFixedWidth(60)
        self.prompt_point_size_spin.valueChanged.connect(self._on_prompt_point_size_changed)
        prompt_point_row_layout.addWidget(self.prompt_point_size_spin)
        prompt_point_row_layout.addStretch()
        self.left_layout.addWidget(prompt_point_row_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_autosegment = QPushButton("Autosegment worms (Frame 1)")
        self.btn_autosegment.clicked.connect(self.run_autosegmentation)
        
        self.btn_track = QPushButton("Track")
        self.btn_track.clicked.connect(self.run_tracking)
        self.btn_track.setEnabled(False) # Disabled until autosegment is done

        self.btn_stop = QPushButton("Stop Tracking")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_tracking)

        # Move tracking buttons into left sidebar
        self.left_layout.addSpacing(10)
        self.left_layout.addWidget(self.btn_autosegment)
        self.left_layout.addWidget(self.btn_track)
        self.left_layout.addWidget(self.btn_stop)

        # Create vertical opacity slider
        opacity_slider_widget = QWidget()
        opacity_slider_layout = QVBoxLayout()
        opacity_slider_layout.setContentsMargins(0, 0, 0, 0)
        opacity_slider_layout.addWidget(QLabel("Mask\nOpacity"), alignment=Qt.AlignHCenter)
        self.mask_opacity_slider = QSlider(Qt.Vertical)
        self.mask_opacity_slider.setMinimum(0)
        self.mask_opacity_slider.setMaximum(100)
        self.mask_opacity_slider.setValue(50)  # default 50%
        self.mask_opacity_slider.setFixedWidth(40)
        self.mask_opacity_slider.valueChanged.connect(self._on_mask_opacity_changed)
        opacity_slider_layout.addWidget(self.mask_opacity_slider, stretch=1)
        self.mask_opacity_label = QLabel("50%")
        self.mask_opacity_label.setAlignment(Qt.AlignHCenter)
        opacity_slider_layout.addWidget(self.mask_opacity_label)
        opacity_slider_widget.setLayout(opacity_slider_layout)

        top_hlayout = QHBoxLayout()
        top_hlayout.addWidget(self.left_sidebar)
        top_hlayout.addWidget(self.image_label, stretch=1)
        top_hlayout.addWidget(opacity_slider_widget)
        top_hlayout.addWidget(self.sidebar_area)
        layout.addLayout(top_hlayout)

        # Controls
        controls_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.frame_label = QLabel("Frame: ")
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.slider)

        # Frame number spinbox for jumping to specific frame
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_spinbox.setValue(0)
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setFixedWidth(60)
        # Connect to slider to keep them in sync
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        controls_layout.addWidget(self.frame_spinbox)

        # Play / Pause button next to the slider
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedSize(80, 24)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        # Model size selector
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "large"])
        # default to 'base'
        self.model_combo.setCurrentText("base")
        self.model_combo.currentTextChanged.connect(lambda v: setattr(self, 'model_size', v))
        controls_layout.addWidget(self.model_combo)

        layout.addLayout(controls_layout)
        # Status Bar / Progress
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # object buttons state
        self.object_buttons = []
        self.selected_object_idx = None
        self.selected_prompt_mode = 1  # 1 for positive (+), 0 for negative (-)

        self.tracking_widget.setLayout(layout)
        self.stacked_widget.addWidget(self.tracking_widget)

    # --- Frame Navigation and Controls ---

    def choose_mask_output_dir(self):
        """Open dialog to choose mask output directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select Mask Output Directory", self.save_mask_dir)
        if folder:
            self.save_mask_dir = folder
            self.mask_dir_label.setText(folder)
            self.status_label.setText(f"Mask output directory set to: {folder}")

    # --- Frame Navigation and Controls ---

    def choose_folder(self):
        """Open directory dialog to select and load a video folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if folder:
            self.video_path = folder
            # Load images
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            # Try to sort numerically if filenames are integers
            try:
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                files.sort()
            
            self.image_files = [os.path.join(folder, f) for f in files]

            if not self.image_files:
                QMessageBox.warning(self, "Error", "No image files found in directory.")
                return

            self.slider.setMaximum(len(self.image_files) - 1)
            self.slider.setEnabled(True)
            self.frame_spinbox.setMaximum(len(self.image_files) - 1)
            self.frame_spinbox.setEnabled(True)
            # enable play button now that frames are available
            try:
                self.btn_play.setEnabled(True)
            except Exception:
                pass
            self.stacked_widget.setCurrentIndex(1)
            # update loader references and clear caches
            try:
                self._image_loader.set_references(self.image_files, self.tracking_mask_files, gui_instance=self)
            except Exception:
                pass
            self.clear_pixmap_cache()
            self.update_display()
            # clear caches when new folder chosen
            self.clear_pixmap_cache()
            # update left-sidebar spinboxes if present
            try:
                n_frames = len(self.image_files)
                if hasattr(self, 'end_spin'):
                    self.end_spin.setMaximum(max(0, n_frames - 1))
                    self.end_spin.setValue(max(0, n_frames - 1))
                if hasattr(self, 'start_spin'):
                    self.start_spin.setMaximum(max(0, n_frames - 1))
                    self.start_spin.setValue(0)
            except Exception:
                pass

    def on_slider_change(self, value):
        """Handle frame slider value changes to navigate between frames."""
        self.current_frame_idx = value
        self.frame_label.setText(f"Frame: {value}")
        # Sync spinbox without triggering its valueChanged signal
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(value)
        self.frame_spinbox.blockSignals(False)
        self.update_display()
        # refresh object sidebar to show prompts for the newly-selected frame
        try:
            self.update_object_sidebar()
        except Exception:
            pass
        # request prefetch of neighbors
        self._prefetch_neighbors(self.current_frame_idx)

    def _on_frame_spinbox_changed(self, value):
        """Handle direct input to frame spinbox to jump to that frame."""
        # Set slider value, which will trigger on_slider_change
        self.slider.setValue(value)

    def _on_mask_opacity_changed(self, value):
        """Handle mask opacity slider changes."""
        self.mask_opacity = value / 100.0
        self.mask_opacity_label.setText(f"{value}%")
        # Clear pixmap cache so masks get redrawn with new opacity
        self.clear_pixmap_cache()
        # Request reload of current frame to show new opacity immediately
        try:
            self._image_loader.request(self.current_frame_idx)
        except Exception:
            pass

    def _on_show_centerlines_changed(self, state):
        """Handle show centerlines checkbox changes."""
        # Clear pixmap cache so centerlines get redrawn/hidden
        self.clear_pixmap_cache()
        # Request reload of current frame
        try:
            self._image_loader.request(self.current_frame_idx)
        except Exception:
            pass

    def _on_centerline_points_changed(self, value):
        """Handle centerline points spinbox changes - resample and redraw."""
        # Resample all existing centerlines to the new number of points
        try:
            for frame_idx in self.centerlines.keys():
                frame_centerlines = self.centerlines[frame_idx]
                updated_centerlines = {}
                for obj_id, centerline_data in frame_centerlines.items():
                    if isinstance(centerline_data, tuple) and len(centerline_data) == 2:
                        full_skeleton, _ = centerline_data
                        # Resample the full skeleton to the new number of points
                        if len(full_skeleton) > 1:
                            resampled = self._resample_centerline(full_skeleton, value)
                        else:
                            resampled = full_skeleton
                        updated_centerlines[obj_id] = (full_skeleton, resampled)
                    else:
                        updated_centerlines[obj_id] = centerline_data
                self.centerlines[frame_idx] = updated_centerlines
            
            # Clear cache and reload current frame
            self.clear_pixmap_cache()
            self._image_loader.request(self.current_frame_idx)
        except Exception:
            pass

    def _on_prompt_point_size_changed(self, value):
        """Handle prompt point size spinbox changes."""
        self.prompt_point_size = value
        # Clear pixmap cache so prompts get redrawn with new size
        self.clear_pixmap_cache()
        # Request reload of current frame to show new point size immediately
        try:
            self._image_loader.request(self.current_frame_idx)
        except Exception:
            pass
        # Update button highlight border width if an object is currently selected
        if self.selected_object_idx is not None:
            try:
                self.on_object_button_clicked(self.selected_object_idx, self.selected_prompt_mode)
            except Exception:
                pass

    def _resample_centerline(self, centerline, num_points):
        """Resample centerline to desired number of points (same logic as TrackerWorker)."""
        if len(centerline) <= 1:
            return centerline
        N = len(centerline)
        if N <= num_points:
            return centerline
        else:
            indices = np.round(np.linspace(0, N-1, num_points)).astype(int)
            resampled = [centerline[j] for j in indices]
            return resampled

    def clear_pixmap_cache(self):
        """Clear the pixmap cache to force reloading of frames."""
        try:
            self.pixmap_cache.clear()
        except Exception:
            self.pixmap_cache = {}

    def run_autosegmentation(self):
        """Run automatic segmentation to detect worms in the current frame."""
        if not self.image_files:
            return

        self.status_label.setText("Autosegmenting...")
        QApplication.processEvents() # Force UI update

        # Run autoseg on the currently selected frame (so prompts appear where you are)
        target_idx = self.current_frame_idx
        img_path = self.image_files[target_idx]
        img = cv2.imread(img_path)
        
        # Run user's logic
        try:
            prompts = process_image_and_scale_centers(img, downsample_resolution=8, num_skeleton_points=10)
            # store prompts as separate objects (one per detected worm)
            # wrap each coordinate as ((x, y), 1) for positive prompts
            for i, skeleton_pts in enumerate(prompts):
                labeled_prompts = [((x, y), 1) for x, y in skeleton_pts]
                self.create_new_object(target_idx, labeled_prompts)
            # mark which frame these prompts belong to
            self.autoseg_frame_idx = target_idx

            # Update the object sidebar with buttons for each detected object (for current frame)
            try:
                self.update_object_sidebar()
            except Exception:
                pass

            all_objs = self.get_all_object_ids()
            self.status_label.setText(f"Found {len(all_objs)} worms. Ready to Track.")
            self.btn_track.setEnabled(True if len(all_objs) > 0 else False)

            # Remove any cached pixmap for that frame (it may have been created before autoseg)
            try:
                if target_idx in self.pixmap_cache:
                    self.pixmap_cache.pop(target_idx, None)
            except Exception:
                pass

            # --- IMMEDIATE DRAW: render prompt points onto the just-segmented frame and display it
            try:
                # draw on a copy (BGR) so we can show markers immediately without waiting for the loader
                img_bgr = img.copy()
                circle_radius = 15
                x_size = 18
                # bright red in BGR for selection highlight
                gold_bgr = (0, 0, 255)
                text_bgr = (255, 255, 255)
                outline_bgr = (0, 0, 0)

                # Iterate over all objects that have prompts in target frame
                for obj_id in self.get_all_object_ids():
                    skeleton_pts = self.get_prompts_for_object_in_frame(obj_id, target_idx)
                    if not skeleton_pts:
                        continue
                    color_rgb = tuple(map(int, self.colors[(obj_id + 1) % len(self.colors)]))
                    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                    is_selected = (self.selected_object_idx is not None and self.selected_object_idx == obj_id)
                    
                    for ((cx, cy), label) in skeleton_pts:
                        cx_i = int(cx)
                        cy_i = int(cy)
                        
                        if label == 1:
                            # Positive prompt: draw circle
                            if is_selected:
                                # Selection outline scales proportionally with point size
                                sel_radius = int(circle_radius * 1.8)
                                inner_radius = int(circle_radius * 1.3)
                                cv2.circle(img_bgr, (cx_i, cy_i), sel_radius, gold_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, color_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), inner_radius, outline_bgr, 4)
                                cv2.putText(img_bgr, str(obj_id+1), (cx_i+inner_radius+2, cy_i-inner_radius-2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                            else:
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, color_bgr, -1)
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, outline_bgr, 1)
                                cv2.putText(img_bgr, str(obj_id+1), (cx_i+8, cy_i-8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
                        else:
                            # Negative prompt: draw X
                            if is_selected:
                                x_half = int(x_size * 1.4)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i - x_half), (cx_i + x_half, cy_i + x_half), gold_bgr, 8)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i + x_half), (cx_i + x_half, cy_i - x_half), gold_bgr, 8)
                                cv2.line(img_bgr, (cx_i - x_half + 4, cy_i - x_half + 4), (cx_i + x_half - 4, cy_i + x_half - 4), color_bgr, 6)
                                cv2.line(img_bgr, (cx_i - x_half + 4, cy_i + x_half - 4), (cx_i + x_half - 4, cy_i - x_half + 4), color_bgr, 6)
                                cv2.putText(img_bgr, str(obj_id+1), (cx_i+x_half+2, cy_i-x_half-2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                            else:
                                x_half = x_size // 2
                                cv2.line(img_bgr, (cx_i - x_half, cy_i - x_half), (cx_i + x_half, cy_i + x_half), color_bgr, 4)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i + x_half), (cx_i + x_half, cy_i - x_half), color_bgr, 4)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i - x_half), (cx_i + x_half, cy_i + x_half), outline_bgr, 1)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i + x_half), (cx_i + x_half, cy_i - x_half), outline_bgr, 1)
                                cv2.putText(img_bgr, str(obj_id+1), (cx_i+x_half+2, cy_i-x_half-2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)

                # convert back to RGB for Qt
                img_rgb_now = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb_now = np.ascontiguousarray(img_rgb_now)
                h, w, ch = img_rgb_now.shape
                bytes_per_line = ch * w
                q_img_now = QImage(img_rgb_now.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap_now = QPixmap.fromImage(q_img_now)
                scaled_now = pixmap_now.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # cache and show immediately
                try:
                    if len(self.pixmap_cache) >= self.pixmap_cache_max:
                        try:
                            self.pixmap_cache.popitem(last=False)
                        except Exception:
                            first_key = next(iter(self.pixmap_cache))
                            self.pixmap_cache.pop(first_key, None)
                except Exception:
                    pass
                self.pixmap_cache[target_idx] = scaled_now
                try:
                    self.pixmap_cache.move_to_end(target_idx, last=True)
                except Exception:
                    pass
                self.image_label.setPixmap(scaled_now)
            except Exception:
                # fallback: request loader as before
                try:
                    self._image_loader.request(target_idx)
                except Exception:
                    pass

            # ensure slider shows that frame and request the loader to (re)load it (for masks/prefetch)
            try:
                self.slider.setValue(target_idx)
                self._image_loader.request(target_idx)
                self._prefetch_neighbors(target_idx)
            except Exception:
                pass

            # update display (will either use cache or wait for loader)
            self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")
            self.status_label.setText("Error in segmentation.")

    # --- Tracking Operations ---
    
    def run_tracking(self):
        """Start SAM2 tracking on the selected frame range."""
        if not SAM2_AVAILABLE:
            QMessageBox.critical(self, "Error", "SAM 2 library is not installed/importable.")
            return

        # If dataset was loaded from HDF5, export frames to a temporary folder first
        if getattr(self, 'h5_file_path', None) is not None:
            # check that prompts exist for requested start frame before exporting
            try:
                start_frame = int(self.start_spin.value()) if hasattr(self, 'start_spin') else 0
            except Exception:
                start_frame = 0
            # Check if any object has prompts in the start frame
            start_frame_has_prompts = any(
                len(self.get_prompts_for_object_in_frame(obj_id, start_frame)) > 0 
                for obj_id in self.get_all_object_ids()
            )
            if not start_frame_has_prompts:
                QMessageBox.critical(self, "Error", "No prompts for chosen start frame. Please add some prompts.")
                return
            
            try:
                end_frame = int(self.end_spin.value()) if hasattr(self, 'end_spin') else None
            except Exception:
                end_frame = None
            

            # start export worker
            try:
                self.btn_track.setEnabled(False)
                self.btn_autosegment.setEnabled(False)
                self.status_label.setText("Exporting HDF5 frames to temporary folder...")
                self._export_worker = ExportH5Worker(self.h5_file_path, start_frame, end_frame, getattr(self, 'h5_dataset_name', None))
                self._export_worker.stream_mode = self.stream_mode_checkbox.isChecked()
                self._export_worker.stream_threshold = self.stream_threshold_spin.value()
                # show progress
                self._export_worker.progress.connect(lambda p: self.status_label.setText(f"Exporting frames... {p}%"))
                # when stream ready, start tracker (but keep exporting remaining frames)
                self._export_worker.ready.connect(self._on_h5_export_ready)
                # when finished, still call on_tracking_finished so masks can be indexed
                # self._export_worker.finished.connect(self.on_tracking_finished)
                self._export_worker.error.connect(self.on_tracking_error)
                self._export_worker.start()
                return
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start HDF5 export: {e}")
                return

        # determine requested frame range from spinboxes (if present)
        try:
            start_frame = int(self.start_spin.value()) if hasattr(self, 'start_spin') else 0
            end_frame = int(self.end_spin.value()) if hasattr(self, 'end_spin') else None
        except Exception:
            start_frame = 0
            end_frame = None

        # normal folder-based tracking
        self._start_tracker_worker(self.video_path, start_frame=start_frame, end_frame=end_frame)

    def stop_tracking(self):
        """Request an early stop for ongoing tracking."""
        try:
            if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
                self.worker.request_stop()
                self.status_label.setText("Stop requested... finishing current frame.")
            try:
                self.btn_stop.setEnabled(False)
            except Exception:
                pass
        except Exception:
            pass

    def _start_tracker_worker(self, video_folder, start_frame=0, end_frame=None):
        """Start the TrackerWorker using the given video_folder path."""
        try:
            self.btn_track.setEnabled(False)
            self.btn_autosegment.setEnabled(False)
            self.status_label.setText("Tracking in progress... This may take a moment.")
            print(f"Starting tracker for frames {start_frame} to {end_frame}")
            # fetch prompts for all objects in the requested start frame
            prompts = self.get_prompts_for_frame(start_frame)
            if not any(prompts):  # Check if any object has prompts
                QMessageBox.critical(self, "Error", "No prompts for chosen start frame. Please add some prompts.")
                self.btn_track.setEnabled(True)
                self.btn_autosegment.setEnabled(True)
                return

            video_name = getattr(self, 'h5_file_path', None) or self.video_path
            print(f"Video name for tracker: {video_name}")
            print(f"trackerworker video folder: {video_folder}")
            print(f"trackerworker frame range: {start_frame} to {end_frame}")
            self.worker = TrackerWorker(video_folder, self.device, prompts, colors=self.colors, model_size=getattr(self, 'model_size', 'base'), start_frame=start_frame, end_frame=end_frame, video_name=video_name, num_centerline_points=self.centerline_points_spin.value(), save_mask_dir=self.save_mask_dir)
            self.worker.maskSaved.connect(self._on_mask_saved)
            self.worker.centerlineComputed.connect(self._on_centerline_computed)
            self.worker.finished.connect(self.on_tracking_finished)
            self.worker.error.connect(self.on_tracking_error)
            self.worker.start()
            try:
                self.btn_stop.setEnabled(True)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracker: {e}")
            self.btn_track.setEnabled(True)
            self.btn_autosegment.setEnabled(True)

    # --- Playback Controls ---

    def _prefetch_neighbors(self, idx):
        """Prefetch frames around the given index for smoother navigation."""
        # Request loading of current frame and neighbors
        start = max(0, idx - self.prefetch_radius)
        end = min(self.slider.maximum(), idx + self.prefetch_radius)
        for i in range(start, end + 1):
            if i not in self.pixmap_cache:
                self._image_loader.request(i)

    def _on_frame_loaded(self, idx, img_rgb):
        """Handle frame loaded signal from background loader thread."""
        # Called in GUI thread when image loader has RGB numpy image ready
        try:
            img_rgb = np.ascontiguousarray(img_rgb)
            # keep a copy of the full-resolution RGB for coordinate mapping (when showing current frame)
            if idx == self.current_frame_idx:
                try:
                    self._last_loaded_rgb = img_rgb.copy()
                except Exception:
                    self._last_loaded_rgb = None
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Draw prompt markers for any frame that has prompts
            all_obj_ids = self.get_all_object_ids()
            has_prompts = any(len(self.get_prompts_for_object_in_frame(obj_id, idx)) > 0 for obj_id in all_obj_ids)
            
            if has_prompts:
                # Draw filled circles for positive and X markers for negative prompts
                # Convert RGB->BGR for OpenCV drawing, then back to RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                circle_radius = self.prompt_point_size
                x_size = self.prompt_point_size + 3
                # bright red in BGR for selection highlight
                gold_bgr = (0, 0, 255)
                text_bgr = (255, 255, 255)
                outline_bgr = (0, 0, 0)

                for obj_id in all_obj_ids:
                    skeleton_pts = self.get_prompts_for_object_in_frame(obj_id, idx)
                    if not skeleton_pts:
                        continue
                    try:
                        is_selected = (self.selected_object_idx is not None and self.selected_object_idx == obj_id)
                        color_rgb = tuple(map(int, self.colors[(obj_id + 1) % len(self.colors)]))
                        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                        for ((cx, cy), label) in skeleton_pts:
                            cx_i = int(cx)
                            cy_i = int(cy)
                            
                            if label == 1:
                                # Positive prompt: draw circle
                                # Draw filled circle with color
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, color_bgr, -1)
                                # Draw outline (gold if selected, black if not)
                                outline_color = gold_bgr if is_selected else outline_bgr
                                outline_thickness = 2 if is_selected else 1
                                cv2.circle(img_bgr, (cx_i, cy_i), circle_radius, outline_color, outline_thickness)
                                # Draw label
                                try:
                                    if is_selected:
                                        cv2.putText(img_bgr, str(obj_id+1), (cx_i+circle_radius+2, cy_i-circle_radius-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                                    else:
                                        cv2.putText(img_bgr, str(obj_id+1), (cx_i+8, cy_i-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
                                except Exception:
                                    pass
                            else:
                                # Negative prompt: draw X
                                x_half = x_size // 2
                                # Draw X lines with color
                                cv2.line(img_bgr, (cx_i - x_half, cy_i - x_half), (cx_i + x_half, cy_i + x_half), color_bgr, 4)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i + x_half), (cx_i + x_half, cy_i - x_half), color_bgr, 4)
                                # Draw outline (gold if selected, black if not)
                                outline_color = gold_bgr if is_selected else outline_bgr
                                outline_thickness = 2 if is_selected else 1
                                cv2.line(img_bgr, (cx_i - x_half, cy_i - x_half), (cx_i + x_half, cy_i + x_half), outline_color, outline_thickness)
                                cv2.line(img_bgr, (cx_i - x_half, cy_i + x_half), (cx_i + x_half, cy_i - x_half), outline_color, outline_thickness)
                                # Draw label
                                try:
                                    if is_selected:
                                        cv2.putText(img_bgr, str(obj_id+1), (cx_i+x_half+2, cy_i-x_half-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_bgr, 3, cv2.LINE_AA)
                                    else:
                                        cv2.putText(img_bgr, str(obj_id+1), (cx_i+x_half+2, cy_i-x_half-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
                                except Exception:
                                    pass
                    except Exception:
                        continue
                # convert back to RGB for Qt
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = np.ascontiguousarray(img_rgb)
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Draw centerlines if available for this frame and checkbox is checked
            show_centerlines = getattr(self, 'show_centerlines_checkbox', None)
            if idx in self.centerlines and show_centerlines and show_centerlines.isChecked():
                # Convert back to BGR for OpenCV drawing
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                centerline_dict = self.centerlines[idx]
                pink_bgr = (255, 192, 203)  # Pink in BGR
                red_bgr = (0, 0, 255)  # Red in BGR
                for obj_id, centerline_data in centerline_dict.items():
                    # centerline_data is (full_skeleton, resampled_points)
                    if isinstance(centerline_data, tuple) and len(centerline_data) == 2:
                        full_skeleton, resampled_points = centerline_data
                    else:
                        # Backward compatibility: if it's just a list, treat as resampled
                        full_skeleton = centerline_data if centerline_data else []
                        resampled_points = centerline_data if centerline_data else []
                    
                    # Draw full skeleton as pink circles (small)
                    for pt in full_skeleton:
                        try:
                            cv2.circle(img_bgr, pt, 1, pink_bgr, -1)
                        except Exception:
                            pass
                    
                    # Draw resampled points as red circles (larger)
                    for pt in resampled_points:
                        try:
                            cv2.circle(img_bgr, pt, 3, red_bgr, -1)
                        except Exception:
                            pass
                # Convert back to RGB
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = np.ascontiguousarray(img_rgb)
                q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # cache scaled pixmap
            try:
                    if len(self.pixmap_cache) >= self.pixmap_cache_max:
                        try:
                            # pop least-recently-used (first item)
                            self.pixmap_cache.popitem(last=False)
                        except Exception:
                            # fallback: clear one item
                            first_key = next(iter(self.pixmap_cache))
                            self.pixmap_cache.pop(first_key, None)
                    self.pixmap_cache[idx] = scaled_pixmap
                    try:
                        # mark as recently used
                        self.pixmap_cache.move_to_end(idx, last=True)
                    except Exception:
                        pass
            except Exception:
                self.pixmap_cache[idx] = scaled_pixmap

            # if this is the currently visible frame, display it
            if idx == self.current_frame_idx:
                self.image_label.setPixmap(scaled_pixmap)
        except Exception:
            pass

    
    def toggle_play(self):
        """Toggle playback: start or stop the play timer."""
        if not self.slider.isEnabled():
            return

        if self.playing:
            # Stop playback
            self.play_timer.stop()
            self.playing = False
            try:
                self.btn_play.setText("Play")
            except Exception:
                pass
        else:
            # Start playback
            # If already at last frame, jump to current to end
            if self.current_frame_idx >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self.playing = True
            try:
                self.btn_play.setText("Pause")
            except Exception:
                pass
            self.play_timer.start()

    def _on_play_timeout(self):
        """Advance one frame on each timer timeout; stop at the end."""
        if self.current_frame_idx < self.slider.maximum():
            # advance by 1 using slider so signals/update_display happen
            self.slider.setValue(self.current_frame_idx + 1)
        else:
            # reached end; stop playback
            self.play_timer.stop()
            self.playing = False
            try:
                self.btn_play.setText("Play")
            except Exception:
                pass

    def keyPressEvent(self, event):
        """Handle left/right arrow keys to navigate frames."""
        key = event.key()
        if key == Qt.Key_Right:
            # move forward one frame
            if self.current_frame_idx < self.slider.maximum():
                self.slider.setValue(self.current_frame_idx + 1)
        elif key == Qt.Key_Left:
            # move back one frame
            if self.current_frame_idx > self.slider.minimum():
                self.slider.setValue(self.current_frame_idx - 1)
        else:
            # pass other keys to base class
            super().keyPressEvent(event)

    # --- Tracking Callbacks ---
    
    def _on_mask_saved(self, frame_idx, mask_path):
        """Called when a single mask is saved during tracking.
        Update mask index and navigate to the frame to show the new mask.
        """
        try:
            # Add to mask index
            self.tracking_mask_files[frame_idx] = mask_path
            # Clear cached mask for this frame
            self._mask_cache.pop(frame_idx, None)
            # Clear cached pixmap for this frame so it reloads with mask
            self.pixmap_cache.pop(frame_idx, None)
            
            # Update image loader references
            self._image_loader.set_references(
                image_files=self.image_files, 
                mask_files=self.tracking_mask_files, 
                h5_path=getattr(self, 'h5_file_path', None), 
                h5_dataset=getattr(self, 'h5_dataset_name', None),
                gui_instance=self
            )
            
            # Navigate to this frame and display the new mask
            self.slider.setValue(frame_idx)
        except Exception:
            pass
    
    def _on_centerline_computed(self, frame_idx, centerline_dict):
        """Called when centerlines are computed for a frame.
        Store centerlines and trigger redraw if viewing this frame.
        """
        try:
            # Store centerlines for this frame
            self.centerlines[frame_idx] = centerline_dict
            # Clear cached pixmap so centerlines get drawn
            self.pixmap_cache.pop(frame_idx, None)
            # If viewing this frame, request reload to draw centerlines
            if frame_idx == self.current_frame_idx:
                self._image_loader.request(frame_idx)
        except Exception:
            pass

    def on_tracking_finished(self, results):
        """`results` is the mask output folder path emitted by the worker.
        Build an index of available mask JPGs and prepare cache structures for fast loading.
        """
        mask_folder = results
        # Merge new masks into the existing map so earlier tracking results stay visible.
        new_masks = {}

        if os.path.isdir(mask_folder):
            for fname in os.listdir(mask_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name, _ = os.path.splitext(fname)
                    try:
                        idx = int(name)
                    except ValueError:
                        continue
                    new_masks[idx] = os.path.join(mask_folder, fname)

        # Drop any cached masks for frames that were just regenerated, then merge.
        for idx in new_masks.keys():
            try:
                self._mask_cache.pop(idx, None)
            except Exception:
                pass
        self.tracking_mask_files.update(new_masks)

        # update image loader references to include mask files
        try:
            self._image_loader.set_references(image_files=self.image_files, mask_files=self.tracking_mask_files, h5_path=getattr(self, 'h5_file_path', None), h5_dataset=getattr(self, 'h5_dataset_name', None), gui_instance=self)
        except Exception:
            pass

        # Mark results available
        self.tracking_results = True
        self.status_label.setText("Tracking Complete.")
        self.btn_track.setEnabled(True)
        self.btn_autosegment.setEnabled(True)
        try:
            self.btn_stop.setEnabled(False)
        except Exception:
            pass
        # Clear any cached scaled pixmaps (they were created without masks)
        try:
            self.pixmap_cache.clear()
        except Exception:
            self.pixmap_cache = {}

        # Prefetch current frame (so the masked version loads) and update display
        try:
            self._image_loader.request(self.current_frame_idx)
            self._prefetch_neighbors(self.current_frame_idx)
        except Exception:
            pass

        self.update_display()

    # --- Event Handling ---
    
    def eventFilter(self, obj, event):
        """Handle mouse events on the image label for prompt interaction."""
        # handle hover and clicks on the image label
        try:
            if obj == self.image_label:
                # Mouse move: detect hover over any plotted prompt point
                if event.type() == QEvent.MouseMove:
                    # reset hover
                    self._hovered_point = None
                    pixmap = self.image_label.pixmap()
                    # get prompts for current frame
                    prompts = self.get_prompts_for_frame(self.current_frame_idx)
                    if pixmap is None or not any(prompts):
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    label_w = self.image_label.width()
                    label_h = self.image_label.height()
                    pixmap_w = pixmap.width()
                    pixmap_h = pixmap.height()
                    offset_x = max(0, (label_w - pixmap_w) // 2)
                    offset_y = max(0, (label_h - pixmap_h) // 2)

                    pos = event.pos()
                    x_in = pos.x() - offset_x
                    y_in = pos.y() - offset_y
                    if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    # get original image size
                    try:
                        if self._last_loaded_rgb is not None:
                            orig_h, orig_w = self._last_loaded_rgb.shape[:2]
                        elif getattr(self, 'h5_file_path', None) is not None:
                            try:
                                with h5py.File(self.h5_file_path, 'r') as f:
                                    ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                    arr = f[ds][self.current_frame_idx]
                                    tmp = np.asarray(arr)
                                    orig_h, orig_w = tmp.shape[:2]
                            except Exception:
                                self.image_label.setCursor(Qt.ArrowCursor)
                                return False
                        else:
                            orig = cv2.imread(self.image_files[self.current_frame_idx])
                            orig_h, orig_w = orig.shape[:2]
                    except Exception:
                        self.image_label.setCursor(Qt.ArrowCursor)
                        return False

                    # compute scale from original->pixmap (preserve aspect ratio so scales equal)
                    scale_x = pixmap_w / float(orig_w)
                    scale_y = pixmap_h / float(orig_h)
                    scale = min(scale_x, scale_y)

                    # threshold in display pixels for selecting a point
                    thr = 12
                    found = False
                    # iterate over all points to find nearest
                    for obj_idx, pts in enumerate(prompts):
                        for pt_idx, ((cx, cy), label) in enumerate(pts):
                            disp_x = int(cx * scale) + offset_x
                            disp_y = int(cy * scale) + offset_y
                            dx = disp_x - pos.x()
                            dy = disp_y - pos.y()
                            if dx*dx + dy*dy <= thr*thr:
                                self._hovered_point = (obj_idx, pt_idx)
                                self.image_label.setCursor(Qt.PointingHandCursor)
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        self.image_label.setCursor(Qt.ArrowCursor)
                    return False

                # Mouse button press: handle different click behaviors
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    # Ctrl+Left: add new object at click (allow on any frame)
                    if (event.modifiers() & Qt.ControlModifier):
                        # allow adding on current frame

                        pixmap = self.image_label.pixmap()
                        if pixmap is None:
                            return False
                        label_w = self.image_label.width()
                        label_h = self.image_label.height()
                        pixmap_w = pixmap.width()
                        pixmap_h = pixmap.height()
                        offset_x = max(0, (label_w - pixmap_w) // 2)
                        offset_y = max(0, (label_h - pixmap_h) // 2)

                        pos = event.pos()
                        x_in = pos.x() - offset_x
                        y_in = pos.y() - offset_y
                        if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                            return True

                        if self._last_loaded_rgb is None:
                            try:
                                if getattr(self, 'h5_file_path', None) is not None:
                                    with h5py.File(self.h5_file_path, 'r') as f:
                                        ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                        arr = f[ds][self.current_frame_idx]
                                        tmp = np.asarray(arr)
                                        orig_h, orig_w = tmp.shape[:2]
                                else:
                                    orig = cv2.imread(self.image_files[self.current_frame_idx])
                                    if orig is None:
                                        return True
                                    orig_h, orig_w = orig.shape[:2]
                            except Exception:
                                return True
                        else:
                            orig_h, orig_w = self._last_loaded_rgb.shape[:2]

                        x_orig = int((x_in * orig_w) / pixmap_w)
                        y_orig = int((y_in * orig_h) / pixmap_h)

                        # create new object with initial prompt at current frame (positive by default)
                        new_obj_id = self.create_new_object(self.current_frame_idx, [((x_orig, y_orig), 1)])
                        self.autoseg_frame_idx = self.current_frame_idx
                        try:
                            self.update_object_sidebar()
                            if self.current_frame_idx in self.pixmap_cache:
                                self.pixmap_cache.pop(self.current_frame_idx, None)
                            self._image_loader.request(self.current_frame_idx)
                            self._prefetch_neighbors(self.current_frame_idx)
                            self.update_display()
                            self.update_display()
                            self.status_label.setText(f"Added object {new_obj_id+1} at ({x_orig},{y_orig})")
                        except Exception:
                            pass
                        return True

                    # Left click without Ctrl: if hovering on a point -> delete that point
                    if self._hovered_point is not None:
                        obj_idx_in_list, pt_idx = self._hovered_point
                        try:
                            all_obj_ids = self.get_all_object_ids()
                            if obj_idx_in_list < len(all_obj_ids):
                                obj_id = all_obj_ids[obj_idx_in_list]
                                prompts_for_obj = self.get_prompts_for_object_in_frame(obj_id, self.current_frame_idx)
                                if 0 <= pt_idx < len(prompts_for_obj):
                                    prompts_for_obj.pop(pt_idx)
                                    self.set_prompts_for_object_in_frame(obj_id, self.current_frame_idx, prompts_for_obj)
                        except Exception:
                            pass

                        # refresh UI (invalidate cache and request recomposition)
                        try:
                            self.update_object_sidebar()
                            if self.current_frame_idx in self.pixmap_cache:
                                self.pixmap_cache.pop(self.current_frame_idx, None)
                            self._image_loader.request(self.current_frame_idx)
                            self._prefetch_neighbors(self.current_frame_idx)
                            self.update_display()
                            self.status_label.setText("Deleted prompt point")
                        except Exception:
                            pass
                        return True

                    # Left click without Ctrl and not on a point: if an object is selected, add a prompt point to that object
                    if self.selected_object_idx is not None:
                        pixmap = self.image_label.pixmap()
                        if pixmap is None:
                            return False
                        label_w = self.image_label.width()
                        label_h = self.image_label.height()
                        pixmap_w = pixmap.width()
                        pixmap_h = pixmap.height()
                        offset_x = max(0, (label_w - pixmap_w) // 2)
                        offset_y = max(0, (label_h - pixmap_h) // 2)

                        pos = event.pos()
                        x_in = pos.x() - offset_x
                        y_in = pos.y() - offset_y
                        if x_in < 0 or y_in < 0 or x_in >= pixmap_w or y_in >= pixmap_h:
                            return True

                        # original image size
                        try:
                            if self._last_loaded_rgb is not None and self.current_frame_idx == self.current_frame_idx:
                                orig_h, orig_w = self._last_loaded_rgb.shape[:2]
                            elif getattr(self, 'h5_file_path', None) is not None:
                                with h5py.File(self.h5_file_path, 'r') as f:
                                    ds = self.h5_dataset_name if getattr(self, 'h5_dataset_name', None) is not None else next(iter(f.keys()))
                                    arr = f[ds][self.current_frame_idx]
                                    tmp = np.asarray(arr)
                                    orig_h, orig_w = tmp.shape[:2]
                            else:
                                orig = cv2.imread(self.image_files[self.current_frame_idx])
                                orig_h, orig_w = orig.shape[:2]
                        except Exception:
                            return True

                        x_orig = int((x_in * orig_w) / pixmap_w)
                        y_orig = int((y_in * orig_h) / pixmap_h)

                        try:
                            # append to selected object's prompt list for current frame with current mode
                            current_prompts = self.get_prompts_for_object_in_frame(self.selected_object_idx, self.current_frame_idx)
                            current_prompts.append(((x_orig, y_orig), self.selected_prompt_mode))
                            self.set_prompts_for_object_in_frame(self.selected_object_idx, self.current_frame_idx, current_prompts)
                        except Exception:
                            # if selected index invalid, ignore
                            return True

                        # refresh UI (invalidate cache and request recomposition)
                        try:
                            self.update_object_sidebar()
                            if self.current_frame_idx in self.pixmap_cache:
                                self.pixmap_cache.pop(self.current_frame_idx, None)
                            self._image_loader.request(self.current_frame_idx)
                            self._prefetch_neighbors(self.current_frame_idx)
                            self.update_display()
                            self.status_label.setText(f"Added prompt to object {self.selected_object_idx+1}")
                        except Exception:
                            pass
                        return True

                # end mouse press handling
        except Exception:
            pass
        return super().eventFilter(obj, event)

    # --- Cleanup ---
    
    def closeEvent(self, event):
        """Clean up resources and stop background threads when closing the application."""
        # Stop background threads cleanly
        try:
            if hasattr(self, '_image_loader') and self._image_loader is not None:
                self._image_loader.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'worker') and getattr(self, 'worker') is not None:
                try:
                    self.worker.terminate()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, '_h5_export_tempdir', None):
                try:
                    shutil.rmtree(self._h5_export_tempdir)
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)

    # --- Error Handling ---
    
    def on_tracking_error(self, err_msg):
        """Handle tracking errors and display error message."""
        QMessageBox.critical(self, "Tracking Error", err_msg)
        self.status_label.setText("Tracking Failed.")
        self.btn_track.setEnabled(True)
        self.btn_autosegment.setEnabled(True)
        try:
            self.btn_stop.setEnabled(False)
        except Exception:
            pass

    # --- HDF5 Export Callbacks ---
    
    def _on_h5_export_finished(self, temp_folder):
        """Called when ExportH5Worker finishes exporting frames to `temp_folder`."""
        try:
            self._h5_export_tempdir = temp_folder
            # determine requested start/end from spinboxes (if present)
            try:
                start_frame = int(self.start_spin.value()) if hasattr(self, 'start_spin') else 0
                end_frame = int(self.end_spin.value()) if hasattr(self, 'end_spin') else None
            except Exception:
                start_frame = 0
                end_frame = None
            # start tracker with the exported folder and requested range
            print(f"starting tracker for frames {start_frame} to {end_frame} after export")
            print(f"temp folder: {temp_folder}")
            self._start_tracker_worker(temp_folder, start_frame=start_frame, end_frame=end_frame)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracker after export: {e}")
            self.btn_track.setEnabled(True)
            self.btn_autosegment.setEnabled(True)

    def _on_h5_export_ready(self, temp_folder):
        """Called when exporter has written an initial batch of frames and the tracker can start."""
        try:
            # keep tempdir reference so we can cleanup later
            self._h5_export_tempdir = temp_folder
            # determine requested start/end from spinboxes (if present)
            try:
                start_frame = int(self.start_spin.value()) if hasattr(self, 'start_spin') else 0
                end_frame = int(self.end_spin.value()) if hasattr(self, 'end_spin') else None
            except Exception:
                start_frame = 0
                end_frame = None
            # start tracker worker now (the export thread will continue writing remaining frames)
            self._start_tracker_worker(temp_folder, start_frame=start_frame, end_frame=end_frame)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracker after HDF5 stream ready: {e}")
            self.btn_track.setEnabled(True)
            self.btn_autosegment.setEnabled(True)

    # --- Object Sidebar Management ---

    def clear_layout(self, layout):
        """Remove all widgets from a layout."""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                try:
                    widget.deleteLater()
                except Exception:
                    pass
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)

    def update_object_sidebar(self):
        """Populate the right-hand sidebar with + and - buttons per object."""
        # Clear existing buttons
        try:
            self.clear_layout(self.object_button_layout)
        except Exception:
            pass

        self.object_buttons = []
        all_object_ids = self.get_all_object_ids()
        
        if not all_object_ids:
            lbl = QLabel("No objects detected")
            self.object_button_layout.addWidget(lbl)
        else:
            # create a row for each object: [+btn] [-btn] [Delete btn]
            for obj_id in all_object_ids:
                # Get prompts for this object in current frame (may be empty)
                prompts_here = self.get_prompts_for_object_in_frame(obj_id, self.current_frame_idx)
                
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 0, 0, 0)

                # Show number of prompts in this frame for this object
                num_prompts = len(prompts_here)
                
                # color chosen in same way as drawing code (use object id +1)
                color_rgb = tuple(map(int, self.colors[(obj_id + 1) % len(self.colors)]))
                r, g, b = color_rgb

                # choose text color based on luminance for readability
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = '#000000' if luminance > 180 else '#FFFFFF'

                # Label showing object name and count
                label_text = f"W{obj_id + 1}"
                if num_prompts > 0:
                    label_text += f" ({num_prompts})"
                obj_label = QLabel(label_text)
                obj_label.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold; font-size: 12px;")
                obj_label.setFixedWidth(50)
                
                # + button for positive prompts
                btn_plus = QPushButton("+")
                btn_plus.setFixedSize(35, 30)
                base_style_plus = f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;"
                btn_plus.setStyleSheet(base_style_plus)
                btn_plus.clicked.connect(lambda _, oid=obj_id: self.on_object_button_clicked(oid, mode=1))

                # - button for negative prompts
                btn_minus = QPushButton("-")
                btn_minus.setFixedSize(35, 30)
                base_style_minus = f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;"
                btn_minus.setStyleSheet(base_style_minus)
                btn_minus.clicked.connect(lambda _, oid=obj_id: self.on_object_button_clicked(oid, mode=0))

                # delete button
                del_btn = QPushButton("✕")
                del_btn.setFixedSize(28, 28)
                del_btn.setStyleSheet("background-color: #ff4d4d; color: #fff; border-radius: 4px;")
                del_btn.clicked.connect(lambda _, oid=obj_id: self.delete_object(oid))

                row_layout.addWidget(obj_label)
                row_layout.addWidget(btn_plus)
                row_layout.addWidget(btn_minus)
                row_layout.addWidget(del_btn)
                row_widget.setLayout(row_layout)

                # Store both buttons for highlighting
                self.object_buttons.append((obj_id, btn_plus, btn_minus))
                self.object_button_layout.addWidget(row_widget)

        # add stretch to push buttons to top
        self.object_button_layout.addStretch(1)
        # enable Track buttons when we have at least one object
        has_objects = len(self.get_all_object_ids()) > 0
        try:
            self.btn_track.setEnabled(has_objects)
        except Exception:
            pass

    def on_object_button_clicked(self, obj_id, mode=1):
        """Handle object button click: mark selected object and mode (+ or -)."""
        self.selected_object_idx = obj_id
        self.selected_prompt_mode = mode  # 1 for positive, 0 for negative
        mode_str = "positive (+)" if mode == 1 else "negative (-)"
        self.status_label.setText(f"Selected object {obj_id+1} - {mode_str} mode")

        # update visual highlight for buttons
        # Scale border width with point size
        border_width = max(1, min(4, self.prompt_point_size // 5))
        print(f"Chosen border width: {border_width}")
        
        for obj_id_stored, btn_plus, btn_minus in self.object_buttons:
            try:
                # recompute base color
                color_rgb = tuple(map(int, self.colors[(obj_id_stored + 1) % len(self.colors)]))
                r, g, b = color_rgb
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = '#000000' if luminance > 180 else '#FFFFFF'
                
                if obj_id_stored == obj_id:
                    # Highlight the selected button with golden border that scales with point size
                    if mode == 1:  # + button selected
                        btn_plus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border: {border_width}px solid #FFD700; border-radius: 6px; padding: 0px; margin: 0px;")
                        btn_minus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;")
                    else:  # - button selected
                        btn_plus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;")
                        btn_minus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border: {border_width}px solid #FFD700; border-radius: 6px; padding: 0px; margin: 0px;")
                else:
                    # Not selected - normal style
                    btn_plus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;")
                    btn_minus.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {text_color}; font-size: 16px; font-weight: bold; border-radius: 6px; padding: 0px; margin: 0px;")
            except Exception as e:
                print(e)
                pass

        # Determine which frame to refresh (use current frame)
        target = self.current_frame_idx

        # Invalidate any existing scaled pixmap for that frame so prompts redraw with selection
        try:
            if target in self.pixmap_cache:
                self.pixmap_cache.pop(target, None)
        except Exception as e:
            print(e)
            pass

        # Request the background loader to reload and redraw this frame with proper selection highlighting
        try:
            self._image_loader.request(target)
            self._prefetch_neighbors(target)
        except Exception as e:
            print(e)
            pass

        """Delete all prompts for object obj_id from all frames."""
        # if obj_id in self.prompts_by_object:
        #     del self.prompts_by_object[obj_id]
        
        # reset selected index if it was the deleted object
        # if self.selected_object_idx == obj_id:
        #     self.selected_object_idx = None

        # # Rebuild sidebar buttons
        # try:
        #     self.update_object_sidebar()
        # except Exception as e:
        #     print(e)
        #     pass

        # # Refresh display so markers reflect deletion
        # try:
        #     # ensure autoseg frame redraw
        #     if self.autoseg_frame_idx is None:
        #         self.autoseg_frame_idx = self.current_frame_idx
        #     # invalidate cached scaled pixmap so update_display won't reuse stale image
        #     try:
        #         if self.current_frame_idx in self.pixmap_cache:
        #             self.pixmap_cache.pop(self.current_frame_idx, None)
        #     except Exception:
        #         pass
        #     self._image_loader.request(self.autoseg_frame_idx)
        #     self._prefetch_neighbors(self.autoseg_frame_idx)
        #     self.update_display()
        # except Exception:
        #     pass

    def delete_object(self, obj_id):
        """Delete all prompts for object obj_id from all frames."""
        try:
            if obj_id in self.prompts_by_object:
                del self.prompts_by_object[obj_id]
            
            # reset selected index if it was the deleted object
            if self.selected_object_idx == obj_id:
                self.selected_object_idx = None

            # Rebuild sidebar buttons
            self.update_object_sidebar()

            # Refresh display so markers reflect deletion
            # invalidate all cached pixmaps for the current frame
            if self.current_frame_idx in self.pixmap_cache:
                self.pixmap_cache.pop(self.current_frame_idx, None)
            
            self._image_loader.request(self.current_frame_idx)
            self._prefetch_neighbors(self.current_frame_idx)
            self.update_display()
            self.status_label.setText(f"Deleted object {obj_id+1}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete object: {e}")
            print(f"delete_object error: {e}")

    # --- Display and Rendering ---
    
    def update_display(self):
        """Update the display to show the current frame."""
        # allow HDF5-backed datasets as well as file lists
        if not self.image_files and not getattr(self, 'h5_file_path', None):
            return
        # If we have a cached scaled pixmap for this frame, use it immediately
        if self.current_frame_idx in self.pixmap_cache:
            try:
                pm = self.pixmap_cache[self.current_frame_idx]
                # update LRU ordering
                try:
                    self.pixmap_cache.move_to_end(self.current_frame_idx, last=True)
                except Exception:
                    pass
                self.image_label.setPixmap(pm)
            except Exception:
                pass
            return

        # Otherwise request the image loader to load this frame (and neighbors via prefetch)
        self._image_loader.request(self.current_frame_idx)
        self._prefetch_neighbors(self.current_frame_idx)

        # Show a lightweight placeholder while loading
        self.image_label.setText("Loading...")
        # Note: actual composition/blending happens in background loader to minimize UI thread work


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = C_Elegans_GUI()
    window.show()
    sys.exit(app.exec_())