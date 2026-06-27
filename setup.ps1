# Stop executing if any command fails
$ErrorActionPreference = "Stop"

# Resolve the repository root to make path handling independent of current working directory.
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
# Path to the local virtual environment folder.
$VENV_DIR = Join-Path $PROJECT_ROOT ".venv"
# Path to the dependency file used for pip installation.
$REQ_FILE = Join-Path $PROJECT_ROOT "requirements.txt"

function Create-Venv {
    # Check if 'python' is available and works
    if (Get-Command python -ErrorAction SilentlyContinue) {
        Write-Host "Attempting to create virtual environment via python -m venv..." -ForegroundColor Cyan
        & python -m venv $VENV_DIR
        if ($LASTEXITCODE -eq 0) { return }
    }

    # Neither approach worked. Print actionable install hints and stop.
    Write-Error "Failed to create virtual environment."
    Write-Host "Please ensure Python is installed and added to your system PATH." -ForegroundColor Red
    Exit 1
}

# Create the venv when it does not exist yet.
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment in: $VENV_DIR" -ForegroundColor Green
    Create-Venv
} else {
    Write-Host "Using existing virtual environment: $VENV_DIR" -ForegroundColor Yellow
}

# Repair case: folder exists but activation script is missing -> incomplete/broken venv.
# Note: Windows venv uses "Scripts\Activate.ps1" instead of "bin/activate"
$ACTIVATE_SCRIPT = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (-not (Test-Path $ACTIVATE_SCRIPT)) {
    Write-Host "Existing virtual environment is incomplete. Recreating: $VENV_DIR" -ForegroundColor Red
    Remove-Item -Recurse -Force $VENV_DIR
    Create-Venv
}

# Activate the virtual environment in this PowerShell process.
& $ACTIVATE_SCRIPT

# Sanity-check that pip is available inside the venv before trying installs.
if (-not (python -m pip --version 2>$null)) {
    Write-Error "The virtual environment exists, but pip is not available inside it."
    Exit 1
}

# Register sys-code as a source root so imports work from any script.
# Windows paths use .venv\Lib\site-packages\
$SITE_PACKAGES = Join-Path $VENV_DIR "Lib\site-packages"
$PTH_FILE = Join-Path $SITE_PACKAGES "sys-code.pth"
$SYS_CODE_PATH = Join-Path $PROJECT_ROOT "sys-code"

# Write the path to the .pth file
Set-Content -Path $PTH_FILE -Value $SYS_CODE_PATH

# Keep pip up to date to reduce install issues with older bundled versions.
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies if requirements.txt exists.
if (Test-Path $REQ_FILE) {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    python -m pip install -r $REQ_FILE
} else {
    Write-Host "No requirements.txt found. Skipping dependency install." -ForegroundColor Yellow
}

# Final usage hint for the user.
Write-Host "`nSetup complete. Activate with: .venv\Scripts\Activate.ps1" -ForegroundColor Green