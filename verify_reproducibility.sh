#!/bin/bash

echo "=========================================="
echo "REPRODUCIBILITY VERIFICATION TEST"
echo "=========================================="

echo ""
echo "1. Running pipeline - RUN 1"
python src/prepare.py > /dev/null 2>&1
python src/train.py > /dev/null 2>&1
cat metrics.json > /tmp/metrics_run1.json
echo "Run 1 metrics saved"

echo ""
echo "2. Cleaning up artifacts"
rm -f model.pkl metrics.json data/processed/train.csv data/processed/test.csv

echo ""
echo "3. Running pipeline - RUN 2"
python src/prepare.py > /dev/null 2>&1
python src/train.py > /dev/null 2>&1
cat metrics.json > /tmp/metrics_run2.json
echo "Run 2 metrics saved"

echo ""
echo "4. Comparing metrics from both runs"
diff /tmp/metrics_run1.json /tmp/metrics_run2.json
if [ $? -eq 0 ]; then
    echo "REPRODUCIBILITY VERIFIED: Both runs produced identical metrics"
else
    echo "ERROR: Metrics differ between runs"
fi
