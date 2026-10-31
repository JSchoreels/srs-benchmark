# Full FSRS-7 re-fits during the stream, users 1-300

`fsrs_refit.py` + `tools/refit-replay` (fsrs-rs's FSRS-7 trainer, `fsrs::benchmark`, same
recency weighting as `--recency`). From the first test review on, FSRS-7 is re-fitted from
scratch on all reviews so far (`review_th` order), daily, every 100 or every 10 reviews.
Each test review is predicted with:

- **newest fit**: the latest fit before the review (review-time prediction, like FSRS7-Instant);
- **scheduling time**: the latest fit made before the card's previous review, i.e. what a
  scheduler had when it set the interval.

Baseline: FSRS-7 with per-fold fsrs-rs fits (`runs/rs-300/FSRS-7.jsonl`). CIs are paired
user-bootstrap 95% intervals on the difference. `python fsrs_refit_compare.py runs/refit-300 runs/rs-300 runs/quick-300`

299 users (user 267 left out: its every-10 run did not finish, see below), 9.64M test reviews, per user:

| | LogLoss | vs per-fold fits | RMSE(bins) | fits per user | fit secs per user |
| --- | --- | --- | --- | --- | --- |
| FSRS-7, per-fold fits (fsrs-rs) | 0.3488 | | 0.0599 | 5 | |
| re-fit daily, newest fit | 0.3387 | -2.9%, [-0.0121, -0.0082] | 0.0526 | 327 | 15.5 |
| re-fit daily, scheduling time | 0.3486 | -0.1%, [-0.0018, +0.0016] | 0.0613 | 327 | 15.5 |
| re-fit every 100, newest fit | 0.3354 | -3.9%, [-0.0159, -0.0114] | 0.0508 | 477 | 32.4 |
| re-fit every 100, scheduling time | 0.3487 | -0.0%, [-0.0018, +0.0016] | 0.0614 | 477 | 32.4 |
| re-fit every 10, newest fit | 0.3320 | -4.8%, [-0.0199, -0.0143] | 0.0495 | 4770 | 344.9 |
| re-fit every 10, scheduling time | 0.3484 | -0.1%, [-0.0022, +0.0013] | 0.0613 | 4770 | 344.9 |
| FSRS7-Instant, at review time | 0.3168 | -9.2%, [-0.0383, -0.0266] | 0.0377 | 5 | |
| FSRS7-Instant, scheduling time (benchmark fits) | 0.3632 | +4.1%, [+0.0108, +0.0186] | 0.0613 | 5 | |

Fit secs are wall time per fit summed; the trainer used about 2 cores per process, so CPU
time may be up to ~2x. Wall time at 8 processes: daily 31 min, every 100 ~45 min, every 10
~5 h 50 min (299 users).

Reading:
- Re-fitting helps at review time, and more often helps more, but even every 10 reviews it
  gets half of FSRS7-Instant's gain (-4.8% vs -9.2%) at ~70x the fit cost.
- At scheduling time re-fitting changes nothing (-0.1%, CI across 0), while FSRS7-Instant is
  worse there (+4.1%). So newer parameters do not help the interval choice; the review-time
  gain comes from information that arrives after the card was scheduled.
- Not run: re-fit before every review (~10x every 10, about 300 CPU-h for these users).

User 267 (118,650 test reviews): every 100 took 66 s of fits; every other user's every-10
run cost 9.7x (median, max 12.3x) its every-100 run, but 267 ran 3.5 h wall / 6.8 CPU-h
before it was stopped at the 20:00 break. To investigate.
