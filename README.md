# SAM2 GUI: C. Elegans Tracker

A graphical user interface (GUI) for tracking C. elegans using the Segment Anything Model 2 (SAM2). This tool enables interactive segmentation, prompt-based tracking, and centerline extraction from video frames or HDF5 datasets, with support for batch processing and visualization.

## Features
- **Interactive GUI** for selecting video folders or HDF5 files
- **Prompt-based segmentation** using positive/negative points
- **Batch tracking** with SAM2 (Segment Anything Model 2)
- **Centerline extraction** and visualization
- **Mask and centerline export**
- **Support for CUDA and Apple MPS acceleration**
- **Customizable tracking parameters**
- **Progress bars and error handling**

## Installation

1. **Clone the repository** and navigate to the `SAM2Gui` directory:
   ```bash
   git clone <repo-url>
   cd SAM2Gui
   ```
2. **Set up a Python environment** (recommended: Python 3.8+):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements_simplified.txt
   # or for full features:
   pip install -r requirements_full.txt
   ```
   Additional requirements may be listed in `requirements.txt` files in subfolders.

4. **Install SAM2** (Segment Anything Model 2):
   - Place the `segment_anything_2` directory in the project root, or install as instructed by the SAM2 repository.

5. **(Optional) Download model checkpoints:**
   - Place SAM2 model checkpoints (e.g., `sam2.1_hiera_base_plus.pt`) in the `checkpoints/` folder.

## Usage

1. **Activate your environment** (if not already):
   ```bash
   source venv/bin/activate
   ```
2. **Run the GUI:**
   ```bash
   python Main.py
   ```
3. **Select a video folder** (containing sequential images like `0.jpg`, `1.jpg`, ...) or an HDF5 file.
4. **Add prompts** (positive/negative points) for objects to segment and track.
5. **Run segmentation/tracking** using the provided buttons.
6. **Export masks and centerlines** as needed.

## Parameters
- Tracking and segmentation parameters (kernel size, iterations, number of prompts, etc.) can be adjusted in the sidebar or config files.
- Output directories for masks and results can be set in the GUI.

## Troubleshooting
- **Missing dependencies:** Ensure all required Python packages are installed. Use the correct requirements file for your system.
- **SAM2 not found:** Make sure the `segment_anything_2` directory is present and properly installed.
- **CUDA/MPS errors:** The application will fall back to CPU if GPU acceleration is unavailable, but performance may be slower.
- **HDF5 import errors:** Ensure `h5py` is installed and the file is not corrupted.

## Folder Structure
- `Main.py` — Main GUI application
- `config.py` — Configuration parameters
- `checkpoints/` — Model weights
- `merge/`, `scripts/` — Utilities and scripts
- `requirements_*.txt` — Dependency lists

## Acknowledgments
- [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything)
- [PyQt5](https://riverbankcomputing.com/software/pyqt/)
- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)

## License
Specify your license here.
