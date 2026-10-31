# FSRS-7 curve candidates: results

Goal: beat per-user FSRS-7 at scheduling time, under the published FSRS-7 protocol
(`--short --secs --equalize_test_with_non_secs`, 5 time-series folds, parameters fixed over
each test fold). Each candidate keeps the FSRS-7 formula and adds at most one parameter.
The baseline is plain FSRS-7 fitted on the same data in the same run, so every comparison
is paired. Code: `fsrs_curve.py`, `models/fsrs_v7_curve.py`, `fsrs_curve_compare.py`.

**Result: none of the three candidates helps.** All LogLoss changes are within ±0.1%.

## Candidates

1. **Distillation from RWKV-Curve.** Training targets become `y' = (1 − a)·y + a·p_teacher`,
   where p_teacher is RWKV-Curve's prediction for that review, made at the card's previous
   review. Scoring stays on the 0/1 outcomes. The teacher was trained on users 5000–10000.
2. **Again count.** The forgetting curve uses `S·(1 + a)^(−w_again)`, where a is the number
   of Again answers in the card's earlier reviews. One parameter, in [−0.5, 2.0].
3. **Sibling starting difficulty.** A new card's initial difficulty is
   `(1 − b)·D0(rating) + b·D_sib`. D_sib is the mean current difficulty of the note's other
   cards, taken from the fold's plain FSRS-7 fit, using only reviews before the new card's
   first review. One parameter b, in [0, 1].

All new parameters start at 0 (plain FSRS-7) and get FSRS-7's L2 pull toward the start,
with sigma 0.5.

**Deleted-card caveat.** RWKV-Curve takes the deleted-card flag as an input, so the
distillation targets may carry information from the future. "Existing cards only" scores
test reviews of cards still in the cards table; for distillation it is the leak-free reading.

## Settings, chosen on 100 dev users

Dev users: 100 random users from 301–4999 with teacher coverage (`random.Random(0)`, file
`runs/curve-dev-users.txt`), 3.30M test reviews. Tables: `curve-dev-table.md`,
`curve-dev-sib-table.md`.

| Scheduling-time LogLoss vs FSRS-7 | per user | review-weighted | RMSE(bins), per user |
| --- | --- | --- | --- |
| distill-0.3 | −0.1% [−0.0009, +0.0005] | −0.0% | +1.6% |
| distill-0.5 | +0.1% [−0.0010, +0.0014] | +0.0% | +3.5% |
| distill-0.7 | +0.3% [−0.0008, +0.0027] | +0.2% | +6.1% |
| again | −0.0% [−0.0001, +0.0000] | −0.0% | −0.0% |
| sibling | −0.0% [−0.0001, −0.0000] | −0.1% [−0.0005, −0.0000] | +0.0% |

Existing cards only gives the same picture. Kept for the quick check: distill-0.3, again,
sibling.

- Fitted w_again: median −0.15 (5–95%: −0.46 to +0.27). The parameter moves, but FSRS-7's
  other parameters already cover what it captures.
- Fitted b: median 0. It is above 0.05 for 34 users (−0.06% LogLoss on them) and above 0.5
  for 13 users (−0.17%). Few cards have a reviewed sibling: the median user has 1%.

## Quick check: users 1–300

The 238 users in 1–300 with teacher coverage (`runs/curve-quick-users.txt`), 7.26M test
reviews, settings fixed above. CIs are paired user-bootstrap 95% intervals on the difference.

| Per user | LogLoss | RMSE(bins) | AUC |
| --- | --- | --- | --- |
| FSRS-7 | 0.3591 | 0.0658 | 0.7143 |
| distill-0.3 | 0.3589 (−0.1%, [−0.0007, +0.0002]) | 0.0665 (+1.2%) | 0.7164 (+0.3%) |
| again | 0.3591 (−0.0%, [−0.0001, −0.0000]) | 0.0657 (−0.1%) | 0.7145 (+0.0%) |
| sibling | 0.3591 (−0.0%, [−0.0001, −0.0000]) | 0.0657 (−0.0%) | 0.7144 (+0.0%) |

| Review-weighted | LogLoss | RMSE(bins) | AUC |
| --- | --- | --- | --- |
| FSRS-7 | 0.3364 | 0.0506 | 0.7148 |
| distill-0.3 | 0.3366 (+0.1%, [−0.0002, +0.0006]) | 0.0513 (+1.5%) | 0.7155 (+0.1%) |
| again | 0.3363 (−0.0%, [−0.0002, −0.0000]) | 0.0504 (−0.4%) | 0.7152 (+0.0%) |
| sibling | 0.3363 (−0.0%, [−0.0001, −0.0000]) | 0.0505 (−0.0%) | 0.7148 (−0.0%) |

Existing cards only, per user: distill-0.3 −0.0% [−0.0006, +0.0004], again −0.0%, sibling
−0.0% [−0.0001, −0.0001]. Full tables: `curve-quick-table.md`.

The again and sibling gains are statistically clear, but they are about 1e-4 LogLoss. No
candidate had a gain big enough to combine, so no combination was run.

## Fit time

| | CPU-h, 238 users | vs FSRS-7 |
| --- | --- | --- |
| FSRS-7 | 3.87 (58.5 s per user) | 1.00x |
| distill-0.3 | 3.86 | 1.00x |
| again | 3.92 | 1.01x |
| sibling | 8.22 | 2.12x (includes the plain FSRS-7 fit it needs for D_sib) |

Wall time: dev 82 + 36 min, quick check 2 h 25 min, all at 8 processes.

## Reading

At scheduling time, per-user FSRS-7 is not limited by these three inputs. Soft targets from
a much better model do not move it either: what RWKV-Curve knows is not something FSRS-7's
34-parameter curve can express per user. The scheduling-time gap to RWKV-Curve (−6.0% on
users 1–4999) needs a change to the model's form, not one added parameter or better targets.
