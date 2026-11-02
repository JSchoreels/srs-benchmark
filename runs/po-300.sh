#!/bin/sh
# FSRS7-Instant, parameter steps only (PR commit 7e47ea6), benchmark protocol (script.py, per-fold fits), users 1-300.
# Waits for the end of the daily break (20:00-01:00), then runs; the compare table goes to runs/po-300.md.
cd "$(dirname "$0")/../../srs-benchmark-po" || exit 1
PY=../srs-benchmark/.venv/Scripts/python.exe
DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k
LOG=../srs-benchmark/runs/po-300.log
log() { echo "[po-300 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
while :; do
  hm=$(date +%H%M)
  if [ "$hm" -ge 0105 ] && [ "$hm" -lt 1930 ]; then break; fi
  sleep 300
done
log "start"
$PY script.py --algo FSRS-7 --instant --short --secs --equalize_test_with_non_secs --max-user-id 300 --processes 6 --data $DATA >> ../srs-benchmark/runs/po-300.out 2>&1
log "script exit $?"
cp result/FSRS-7-instant-short-secs-equalize_test_with_non_secs.jsonl ../srs-benchmark/runs/po-300/FSRS7-Instant.jsonl
cp ../srs-benchmark/runs/quick-300/FSRS-7.jsonl ../srs-benchmark/runs/po-300/FSRS-7.jsonl
cp ../srs-benchmark/runs/sod-300/params-only-review.jsonl ../srs-benchmark/runs/po-300/params-only-fsrs-rs.jsonl
cd ../srs-benchmark && $PY fsrs_instant_compare.py runs/po-300 > runs/po-300.md 2>&1
log "compare exit $? -> runs/po-300.md"
log "done"
