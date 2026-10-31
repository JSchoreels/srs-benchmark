100 users, 2,097,845 test reviews

1. FSRS-7, script.py vs published (per-user mean; max |per-user difference|)

| | LogLoss | RMSE(bins) | AUC |
| --- | --- | --- | --- |
| published | 0.3604 | 0.0656 | 0.7203 |
| script.py | 0.3603 | 0.0656 | 0.7205 |
| max abs diff | 5.1e-03 | 4.2e-03 | 1.0e-02 |

users whose test-review count differs from the published row: 5

2. FSRS7-Instant, Python port (script.py --instant) vs Rust replay, same fold fits

per-user LogLoss: max |diff| 0.0e+00, mean 0.0e+00; per review: max |p diff| 1.6e-06 over 2,097,845 test reviews

3. FSRS7-Instant vs FSRS-7, both from script.py

| Per user (README) | LogLoss | RMSE(bins) | AUC |
| --- | --- | --- | --- |
| FSRS-7 | 0.3603 | 0.0656 | 0.7205 |
| FSRS7-Instant | 0.3242 (-10.0%, [-0.0472, -0.0270]) | 0.0380 (-42.2%, [-0.0341, -0.0220]) | 0.7799 (+8.3%, [+0.0468, +0.0736]) |

| Review-weighted | LogLoss | RMSE(bins) | AUC |
| --- | --- | --- | --- |
| FSRS-7 | 0.3648 | 0.0610 | 0.7235 |
| FSRS7-Instant | 0.3291 (-9.8%, [-0.0474, -0.0249]) | 0.0295 (-51.6%, [-0.0403, -0.0230]) | 0.7907 (+9.3%, [+0.0504, +0.0883]) |

