# Setup script for SAM2 GUI: C. Elegans Tracker
$ErrorActionPreference = "Stop"

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 1. Install uv if not already installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# 2. Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    uv venv .venv
}

Write-Host "Activating virtual environment..."
.venv\Scripts\Activate.ps1

# 1. Clone and install SAM2 if not already present
if (-not (Test-Path "segment_anything_2")) {
    Write-Host "Cloning SAM2..."
    git clone https://github.com/facebookresearch/segment-anything-2 segment_anything_2
}

Write-Host "Installing SAM2..."
uv pip install -q -e segment_anything_2/.

# 3. Install project dependencies
Write-Host "Installing project dependencies..."
uv pip install -q -r requirements.txt

# 3. Download model checkpoints
Write-Host "Downloading model checkpoints..."
New-Item -ItemType Directory -Force -Path "checkpoints" | Out-Null

$baseUrl = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
$models = @(
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt"
)

foreach ($model in $models) {
    $dest = "checkpoints\$model"
    if (-not (Test-Path $dest)) {
        Write-Host "Downloading $model..."
        Invoke-WebRequest -Uri "$baseUrl/$model" -OutFile $dest
    } else {
        Write-Host "$model already exists, skipping."
    }
}

Write-Host "Setup complete. Activate your environment with '.venv\Scripts\Activate.ps1' then run 'python Main.py' to start."
