#!/bin/bash
# Train + sample TabDDPM against the 5 acceptance-criteria datasets, then run
# TSTR evaluation. Uses a lightweight smoke config by default (SMOKE=1) so this
# can run in a few minutes; set SMOKE=0 for a full benchmark-quality run.
set -e

DATASETS=(car magic nursery shuttle adult)
SMOKE="${SMOKE:-1}"

for ds in "${DATASETS[@]}"; do
  echo "=== TabDDPM: ${ds} (smoke=${SMOKE}) ==="
  SMOKE="${SMOKE}" DATASET="${ds}" python scripts/_run_tabddpm_one.py
done
