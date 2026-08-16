#!/bin/bash
# Train + sample + TSTR-evaluate gmm, bayesian_network, tabdiff and tabmt
# against all 5 acceptance-criteria datasets.
set -e

if [ -n "$MODEL" ]; then
  MODELS=("$MODEL")
else
  MODELS=(gmm bayesian_network tabdiff tabmt)
fi
DATASETS=(car magic nursery shuttle adult)
SMOKE="${SMOKE:-1}"

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "=== ${model}: ${ds} (smoke=${SMOKE}) ==="
    SMOKE="${SMOKE}" MODEL="${model}" DATASET="${ds}" PYTHONPATH=. python3 scripts/_run_model_one.py
  done
done
