# Ridge Regression for Time-Series Forecasting

This repository contains my implementation of ridge regression for univariate time-series prediction using three solvers, all in **pure NumPy**:

- **Closed-form (normal equations)**
- **Cholesky decomposition**
- **Conjugate Gradient (CG)**

> Coursework report: see [`paper/hw2-15-17.pdf`](paper/hw2-15-17.pdf)

## Repo Structure

.
├─ src/
│ ├─ linear_regression.py
│ ├─ linear_regression_cholesky.py
│ ├─ linear_regression_just_closed.py
│ └─ linear_regression_just_cg.py
├─ data/
│ ├─ train_series.csv
│ └─ test_series.csv
├─ reports/
│ └─ figures/
├─ results/
├─ requirements.txt
├─ .gitignore
└─ LICENSE


## Environment
- Python 3.10  
- Install:
  ```bash
  pip install -r requirements.txt
