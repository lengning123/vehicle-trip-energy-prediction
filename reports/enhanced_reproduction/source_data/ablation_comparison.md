# Strict baseline vs enhanced ablation

Both models use the same trip-disjoint split; negative RMSE/MAE change means lower error.

## A1 CNN-BDT

| Metric | Strict | Enhanced | Change |
|---|---:|---:|---:|
| rmse_kw | 6.0028 | 5.65033 | -5.872% |
| mae_kw | 3.52816 | 3.37848 | -4.242% |
| corr | 0.856001 | 0.87359 | +0.018 |
| mae_dev_mj_per_trip | 0.604602 | 0.56468 | -6.603% |
| mean_edev_percent | 354.383 | 250.437 | -29.331% |

## A2 CNN7-Covariance

| Metric | Strict | Enhanced | Change |
|---|---:|---:|---:|
| rmse_kw | 11.7885 | 8.43406 | -28.455% |
| mae_kw | 8.04066 | 5.747 | -28.526% |
| corr | 0.00340463 | 0.700339 | +0.697 |
| mae_dev_mj_per_trip | 1.84019 | 1.1295 | -38.620% |
| mean_edev_percent | 3962.79 | 749.645 | -81.083% |
