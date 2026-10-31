#!/bin/sh
# Curve candidates at the dev settings on the 238 users in 1-300 that the teacher covers.
cd "$(dirname "$0")/.."
export FC_OUT=runs/curve-quick FC_USERS=runs/curve-quick-users.txt FC_PROCESSES=${FC_PROCESSES:-8} \
  FC_DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k \
  FC_TEACHER=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k \
  FC_VARIANTS='[{"name":"distill-0.3","distill":0.3},{"name":"again","again":true},{"name":"sibling","sibling":true}]'
../srs-benchmark/.venv/Scripts/python.exe fsrs_curve.py
