# Data

This project trains and evaluates ridge-regression models on a univariate temperature time series.

## Files

```
data/
├─ train_series.csv   # training split
└─ test_series.csv    # test/holdout split
```

* **Format:** CSV, **one numeric column** of temperatures (°C).
* **Header:** allowed (recommended: `temperature`).
* **Index/time column:** not required; samples are assumed to be equally spaced and indexed by row order.

### Minimal example

```csv
temperature
14.2
13.9
15.1
...
```

## Expected schema

| Column        | Type  | Required | Notes                                  |
| ------------- | ----- | -------: | -------------------------------------- |
| `temperature` | float |      Yes | Univariate target used for forecasting |

> If your CSV has a different column name or additional columns, the scripts use the **first column** as the series by default—update the loader path/column selection in `src/*.py` if needed.

## How the code uses these files

* The scripts read `data/train_series.csv` and `data/test_series.csv` directly.
* Windowing/feature construction is performed inside the code (no feature files required).
* No external metadata is needed.

## Replacing the dataset

To use a different time series:

1. Put your two CSVs in this folder with the same filenames:

   * `train_series.csv`
   * `test_series.csv`
2. Ensure they follow the **one-column numeric** format above.
3. Re-run the scripts from the project root:

   ```bash
   python src/linear_regression.py
   # or the solver-specific scripts
   ```

## Large files, versioning, and LFS

* Keep individual files **< 25 MB** for smooth GitHub web uploads.
* For larger datasets, prefer:

  * Git LFS (optional)
  * A small **sample** here plus a link to full data (e.g., cloud storage) in this README.

## Integrity (optional)

If you share data publicly, include checksums:

```bash
# macOS/Linux
shasum -a 256 data/train_series.csv data/test_series.csv
```

Record the output here so others can verify downloads.

## Licensing & provenance

* Include the source and license of the data if it’s not your own collection.
* If redistribution isn’t allowed, provide a download link and keep only a small sample in this folder.

---

