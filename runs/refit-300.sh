#!/bin/sh
# Full FSRS-7 re-fits during the stream (fsrs_refit.py), users 1-300: daily, every 100, every 10 reviews.
cd "$(dirname "$0")/.."
for E in ${FR_SCHEDULES:-0 100 10}; do
  date +%T; echo "every $E"
  FR_OUT=runs/refit-300 FR_USERS=runs/quick-users.txt FR_EVERY=$E FR_PROCESSES=${FR_PROCESSES:-8} \
    FR_REFIT=C:/Users/jonat/workspace/fsrs-rs/tools/instant-replay/target/release/refit-replay.exe \
    FR_DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k \
    .venv/Scripts/python.exe fsrs_refit.py 2>&1 | grep -E "^error|Traceback|^300/|^[0-9]+/[0-9]+ user" | grep -vE "^[0-9]+/[0-9]+ user" ; grep -h "" /dev/null
done
date +%T
