#!/bin/bash
# setup.sh — one-time environment setup for thesis implementation
set -e

echo "=========================================="
echo "  Thesis Implementation — Setup"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ from python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "→ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
echo "→ Activating venv..."
source venv/bin/activate

# Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "→ Installing requirements (this takes 3-5 minutes)..."
pip install -r requirements.txt --quiet

# Verify imports
echo ""
echo "→ Verifying installation..."
python -c "
import numpy, pandas, sklearn, imblearn, xgboost, matplotlib, seaborn, shap, joblib
print(f'  numpy        {numpy.__version__}')
print(f'  pandas       {pandas.__version__}')
print(f'  scikit-learn {sklearn.__version__}')
print(f'  imblearn     {imblearn.__version__}')
print(f'  xgboost      {xgboost.__version__}')
print(f'  shap         {shap.__version__}')
try:
    import tensorflow as tf
    print(f'  tensorflow   {tf.__version__}')
except Exception as e:
    print(f'  tensorflow   FAILED: {e}')
"

# Create output folders
mkdir -p figures models results

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Place edge_iiot.csv in this folder"
echo "  2. Run: source venv/bin/activate"
echo "  3. Run: python 00_check_data.py    (verify column names)"
echo "  4. Run: ./run_all.sh                (full pipeline)"
echo ""
