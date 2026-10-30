#!/bin/sh
# Parity run in upstream's own protocol (script.py, 5 time-series folds, per-user refits), users 1-100,
# from a scratch worktree of the PR branch (published FSRS-7 result file removed there so the users are run).
cd "$(dirname "$0")/../../srs-benchmark-parity" || exit 1
PY=../srs-benchmark/.venv/Scripts/python.exe
DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k
$PY script.py --algo FSRS-7 --short --secs --equalize_test_with_non_secs --max-user-id 100 --processes 8 --data $DATA || exit 1
$PY script.py --algo FSRS-7 --instant --short --secs --equalize_test_with_non_secs --max-user-id 100 --processes 8 --data $DATA --file
