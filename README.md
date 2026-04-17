
An interactive GUI to use SAM2 for segmentation of objects across video. Designed for .hdf5 data
- includes a module to convert folder of tiff images to hdf5
- supports folder of .png images, but slower

## Installation

```bash
git clone https://github.com/huayinl/SAM2Gui.git
cd SAM2Gui
```

Then run the setup script for your platform. It will install `uv`, create a virtual environment, install dependencies, and download model checkpoints automatically.

**macOS / Linux**
```bash
bash setup.sh
```

**Windows**
```powershell
.\setup.ps1
```

# Quick Start
Now, to run the program just activate the environment and run the script:

**macOS / Linux**
```bash
source .venv/bin/activate
python Main.py
```

**Windows**
```powershell
.venv\Scripts\Activate.ps1
python Main.py
```

