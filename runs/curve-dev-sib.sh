#!/bin/sh
# Sibling starting difficulty on the same 100 dev users (random from 301-4999 within the teacher's coverage, seed 0).
cd "$(dirname "$0")/.."
export FC_OUT=runs/curve-dev-sib FC_USERS=runs/curve-dev-users.txt FC_PROCESSES=${FC_PROCESSES:-8} \
  FC_DATA=C:/Users/jonat/workspace/anki-rwkv/data/anki-revlogs-10k \
  FC_TEACHER=C:/Users/jonat/workspace/anki-rwkv/artifacts/cache/kd-replay/fork-fp32-final8k \
  FC_VARIANTS='[{"name":"sibling","sibling":true}]'
../srs-benchmark/.venv/Scripts/python.exe fsrs_curve.py
