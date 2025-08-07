# Binning Impact on Model Performance

Compares how different binning methods affect linear regression performance on Rossmann sales data.

## Code Files

### `sample_n_from_csv.py`
Samples 5000 rows from `merged_raw_data_5000.csv` → `merged_raw_sampled_500.csv`

### `date_handling.py` 
- Creates `merged_raw_sampled_500_date_only.csv` (removes DayOfWeek column)
- Creates `merged_raw_discretized_500_week_month.csv` (converts dates to months)

### `discretize_with_range.py`
Equal-width binning for Customers and CompetitionDistance (10 bins each) → `discretized_raw_sampled_500.csv`

### `discretize_month.py`
Quarterly month binning: Q1(0,3], Q2(3,6], Q3(6,9], Q4(9,12] → `discretized_month_raw_sampled_500.csv`

### `discretize_d3_with_range.py`
Decision tree supervised binning for Customers and CompetitionDistance → `discretized_d3_raw_sampled_500.csv`

### `d3_binning_competition_distance.py`
Decision tree binning only for CompetitionDistance → `merged_raw_discretized_500_week_month_comp_binned.csv`

### `compare_model_data.py`
Trains Linear Regression on original vs binned data. Shows R², MAE, MSE, and top 10 feature coefficients.

## How to Run

```bash
# 1. Sample data
python code/sample_n_from_csv.py

# 2. Preprocess dates
python code/date_handling.py

# 3. Apply binning (choose one or more)
python code/discretize_with_range.py
python code/discretize_month.py
python code/discretize_d3_with_range.py
python code/d3_binning_competition_distance.py

# 4. Compare models
python code/compare_model_data.py
```

## Data Flow

```
merged_raw_data_5000.csv
  ↓ sample_n_from_csv.py
merged_raw_sampled_500.csv
  ↓ date_handling.py
merged_raw_sampled_500_date_only.csv + merged_raw_discretized_500_week_month.csv
  ↓ binning scripts
discretized datasets
  ↓ compare_model_data.py
performance metrics
```
