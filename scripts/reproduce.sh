#!/usr/bin/env bash
# Rebuild everything from scratch: train, quantise, export, build, test,
# score and benchmark. Needs Python 3 with python/requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")/.."

(cd python && python3 train.py && python3 export.py)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
(cd python && python3 evaluate.py | tee ../results/accuracy.md && python3 real_demo.py)
scripts/bench.sh results
