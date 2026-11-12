#!/usr/bin/env bash
# Rebuild everything from scratch: train both models, quantise, export,
# build, test, score and benchmark. Needs Python 3 with python/requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")/.."

# The classifier steps need the Raman dataset in ../raman-data (see
# python/train_classifier.py for where to get it).
python3 python/train.py
python3 python/train_classifier.py
python3 python/export.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 python/evaluate_classifier.py | tee results/classifier_accuracy.md
python3 python/evaluate.py | tee results/accuracy.md
python3 python/real_demo.py
scripts/bench.sh results
