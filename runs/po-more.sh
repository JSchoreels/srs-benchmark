#!/bin/sh
# Params-only FSRS7-Instant (lr 1e-3, no user bias / Again weight) on the PR's other tables:
# default-parameter start (users 1-300), fsrs-rs fits (users 1-300), shared fit (4,999 users).
cd "$(dirname "$0")/.."
log() { echo "[po-more $(date '+%m-%d %H:%M:%S')] $*" >> runs/po-more.log; }
export FI_DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k FI_PROCESSES=6
PO='[{"name": "params-only", "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "lr": 0.001}]'
for d in defaults-300 shared-5k; do mkdir -p runs/$d-po; cp runs/$d/shared.json runs/$d-po/; done
log "defaults-300-po start"
FI_OUT=runs/defaults-300-po FI_USERS=runs/quick-users.txt FI_CONFIGS="$PO" .venv/Scripts/python.exe fsrs_instant_shared.py replay > runs/defaults-300-po.out 2>&1
log "defaults-300-po exit $?"
log "rs-300-po start"
FI_OUT=runs/rs-300-po FI_USERS=runs/quick-users.txt FI_CONFIGS="$PO" FI_FITTER=rs \
  FR_FIT=C:/Users/jonat/workspace/fsrs-rs/tools/instant-replay/target/release/fit-v7.exe \
  .venv/Scripts/python.exe fsrs_instant.py > runs/rs-300-po.out 2>&1
log "rs-300-po exit $?"
log "shared-5k-po start"
FI_OUT=runs/shared-5k-po FI_USERS=runs/rwkv-test-users.txt FI_CONFIGS="$PO" .venv/Scripts/python.exe fsrs_instant_shared.py replay > runs/shared-5k-po.out 2>&1
log "shared-5k-po exit $?"
log done
