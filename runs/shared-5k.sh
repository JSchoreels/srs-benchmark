#!/bin/sh
# Protocol B: shared FSRS-7 fit on 100 of RWKV's training users, then replay its 4,999 test users.
cd "$(dirname "$0")/.."
export FI_OUT=runs/shared-5k FI_PROCESSES=${FI_PROCESSES:-8} FI_DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k
[ -f runs/shared-5k/shared.json ] || FI_TRAIN=runs/shared-train-100.txt .venv/Scripts/python.exe fsrs_instant_shared.py fit || exit 1
FI_USERS=runs/rwkv-test-users.txt FI_CONFIGS="$(cat runs/shared.json)" .venv/Scripts/python.exe fsrs_instant_shared.py replay
