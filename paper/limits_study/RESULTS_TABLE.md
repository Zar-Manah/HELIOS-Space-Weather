# Transit limits table (four cells)

| Cohort | Model | n | MAE (h) | Median AE (h) | RMSE (h) | Bias (h) | Coverage@90 | Mean width (h) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_n126 | served_transit_v3 | 126 | 15.4713 | 12.2499 | 20.7469 | -1.4744 | 0.9206 | 74.0419 |
| full_n126 | bench_transit_v3_bench | 126 | 15.4275 | 11.0394 | 20.8825 | -1.0402 | 0.7857 | 49.2887 |
| frozen_1to1_n91 | served_transit_v3 | 91 | 11.5792 | 8.8489 | 16.1284 | -1.4598 | 0.9560 | 71.4393 |
| frozen_1to1_n91 | bench_transit_v3_bench | 91 | 10.5336 | 6.5440 | 14.7272 | -1.4055 | 0.8791 | 49.1864 |

`bench_transit_v3_bench` is **not served**. It is the artefact that G2.5 certifies.
`served_transit_v3` is the production model (`models/v3/transit_v3.joblib`).

## Paired model contrast (served − bench), 4,000 bootstrap resamples

| Cohort | ΔMAE (h) | 95 % CI | p (two-sided) |
|---|---:|---|---:|
| full_n126 | +0.0438 | [-0.8367, +0.9117] | 0.9395 |
| frozen_1to1_n91 | +1.0456 | [+0.0967, +1.9837] | 0.0270 |
