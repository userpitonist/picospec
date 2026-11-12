#!/usr/bin/env bash
# Build in Release mode and benchmark every kernel this CPU supports.
# Usage: scripts/bench.sh [output-directory]
set -euo pipefail

cd "$(dirname "$0")/.."
out="${1:-results}"
mkdir -p "$out"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j > /dev/null

{
  echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "os:       $(uname -srm)"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "cpu:      $(sysctl -n machdep.cpu.brand_string)"
  else
    echo "cpu:      $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
  fi
  echo "compiler: $(${CXX:-c++} --version | head -1)"
  echo
  ./build/picospec bench \
      --model models/denoiser_f32.psm \
      --model models/denoiser_int8.psm \
      --len 256 --samples 300 --json "$out/bench.json"
} | tee "$out/bench.md"
