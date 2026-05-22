#!/bin/bash
# run_all.sh — execute the complete pipeline
set -e

# Activate venv if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "ERROR: venv not found. Run ./setup.sh first."
        exit 1
    fi
fi

# Check dataset exists
if [ ! -f "edge_iiot.csv" ]; then
    echo "ERROR: edge_iiot.csv not found in this folder."
    echo "Copy your dataset here, then re-run."
    exit 1
fi

START=$(date +%s)

echo ""
echo "=========================================="
echo "  STAGE 1/4: Training all models"
echo "=========================================="
python3 01_train_save.py

echo ""
echo "=========================================="
echo "  STAGE 2/4: Generating visualizations"
echo "=========================================="
python3 02_visualize.py

echo ""
echo "=========================================="
echo "  STAGE 3/4: SHAP explainability"
echo "=========================================="
python3 03_shap.py

echo ""
echo "=========================================="
echo "  STAGE 4/4: Multi-class analysis"
echo "=========================================="
python3 04_multiclass.py

END=$(date +%s)
RUNTIME=$((END - START))
MINS=$((RUNTIME / 60))
SECS=$((RUNTIME % 60))

echo ""
echo "=========================================="
echo "  PIPELINE COMPLETE in ${MINS}m ${SECS}s"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  → figures/   $(ls figures/ 2>/dev/null | wc -l | xargs) PNG files"
echo "  → results/   $(ls results/ 2>/dev/null | wc -l | xargs) result files"
echo "  → models/    $(ls models/ 2>/dev/null | wc -l | xargs) saved models"
echo ""
echo "Ready for PPT. Check figures/ folder."
