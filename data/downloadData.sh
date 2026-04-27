#!/bin/bash
# download_data.sh
# Downloads SAR-RARP50 dataset from Kaggle
# Requires: kaggle API key at ~/.kaggle/kaggle.json
#
# Usage: bash data/download_data.sh

set -euo pipefail

cd /usr/project/xtmp/acc123
mkdir -p sar-rarp50
cd /usr/project/xtmp/acc123/sar-rarp50

source /usr/project/xtmp/acc123/venv/bin/activate

echo "Downloading test set..."
kaggle datasets download -d umarfrq/sar-rarp50-test-set
unzip sar-rarp50-test-set.zip -d test-set

echo "Downloading train set..."
kaggle datasets download -d umarfrq/sar-rarp50-train-set
unzip sar-rarp50-train-set.zip -d train-set

echo "Done!"
echo "Train set: /usr/project/xtmp/acc123/sar-rarp50/train-set"
echo "Test set:  /usr/project/xtmp/acc123/sar-rarp50/test-set"
echo ""
echo "Next step: run preprocess_sar_rarp50.py"