# Binning Impact on Model Performance

This project analyzes the impact of different binning (discretization) techniques on machine learning model performance using Rossmann store sales data. The project compares the effectiveness of various feature discretization methods and evaluates their impact on predictive model accuracy.

## 📁 Project Structure

```
eda_binning/
├── README.md
├── requirements.txt
├── code/
│   ├── sample_n_from_csv.py
│   ├── discretize_with_range.py
│   ├── discretize_month.py
│   ├── discretize_d3_with_range.py
│   └── compare_model_data.py
└── data/
    ├── merged_raw_data_5000.csv         # Original large dataset
    ├── merged_raw_sampled_500.csv       # Sampled dataset (500 rows)
    ├── discretized_raw_sampled_500.csv  # Equal-width binned data
    ├── discretized_month_raw_sampled_500.csv    # Month binned data
    └── discretized_d3_raw_sampled_500.csv       # Decision tree binned data
```

## 🎯 Project Overview

The project implements and compares three different binning strategies:

1. **Equal-Width Binning**: Divides continuous features into equal-width intervals
2. **Month-Based Binning**: Groups months into quarterly seasons
3. **Supervised Binning**: Uses decision trees to create optimal bins based on target variable

## 📊 Dataset

The project uses Rossmann store sales data with the following features:
- **Store**: Store identifier
- **DayOfWeek**: Day of the week (1-7)
- **Date**: Date of sale
- **Sales**: Sales amount (target variable)
- **Customers**: Number of customers
- **Open**: Whether store was open (0/1)
- **Promo**: Promotion status (0/1)
- **StateHoliday**: State holiday indicator
- **SchoolHoliday**: School holiday indicator (0/1)
- **CompetitionDistance**: Distance to nearest competitor
- **Promo2**: Extended promotion indicator (0/1)

## 🛠️ Installation

1. Clone the repository or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Requirements
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

## 🚀 Usage

### Step 1: Sample Data (Optional)
If you have a large dataset and want to create a smaller sample for faster processing:

```bash
python code/sample_n_from_csv.py
```

**What it does:**
- Samples 500 rows from the original dataset
- Creates `merged_raw_sampled_500.csv` for analysis
- Uses random state for reproducible sampling

### Step 2: Apply Binning Techniques

#### Option A: Equal-Width Binning
```bash
python code/discretize_with_range.py
```

**Features:**
- Divides continuous features into 10 equal-width bins
- Bins `Customers` and `CompetitionDistance` features
- Creates labeled ranges like "(100,200]", "(200,300]"
- Output: `discretized_raw_sampled_500.csv`

#### Option B: Month-Based Binning
```bash
python code/discretize_month.py
```

**Features:**
- Converts dates to months and bins them quarterly
- Quarter 1: Jan-Mar "(0,3]"
- Quarter 2: Apr-Jun "(3,6]"
- Quarter 3: Jul-Sep "(6,9]"
- Quarter 4: Oct-Dec "(9,12]"
- Output: `discretized_month_raw_sampled_500.csv`

#### Option C: Supervised Binning (Decision Tree)
```bash
python code/discretize_d3_with_range.py
```

**Features:**
- Uses Decision Tree Regressor to find optimal split points
- Creates bins based on target variable (Sales) relationships
- Handles missing values automatically
- Falls back to quantile binning if decision tree fails
- Output: `discretized_d3_raw_sampled_500.csv`

### Step 3: Compare Model Performance
```bash
python code/compare_model_data.py
```

**What it does:**
- Loads original and binned datasets
- Preprocesses data (handles missing values, encodes categorical variables)
- Trains Linear Regression models on both datasets
- Compares performance metrics:
  - **R² Score**: Coefficient of determination
  - **MAE**: Mean Absolute Error
  - **MSE**: Mean Squared Error
- Displays top 10 most important features by coefficient magnitude

## 📈 Understanding the Output

When you run `compare_model_data.py`, you'll see output like:

```
=== ORIGINAL DATASET ===
R² Score: 0.8234
MAE: 1205.67
MSE: 2847392.45
Top 10 Features by Coefficient Magnitude:
Customers           2847.32
Store               15.43
DayOfWeek          234.56
...

=== BINNED DATASET ===
R² Score: 0.7891
MAE: 1298.34
MSE: 3124567.89
Top 10 Features by Coefficient Magnitude:
Customers_binned    2234.12
Store               18.76
...
```

### Metrics Interpretation:
- **Higher R² Score**: Better model explanation of variance
- **Lower MAE/MSE**: Better prediction accuracy
- **Feature Importance**: Shows which features most influence predictions

## 🔧 Customization

### Modify Binning Parameters

**Equal-Width Binning:**
```python
# In discretize_with_range.py
main(input_csv, output_csv, num_bins=15)  # Change from 10 to 15 bins
```

**Supervised Binning:**
```python
# In discretize_d3_with_range.py
supervised_binning(feature, target, max_bins=15, min_samples_split=30)
```

### Change Input/Output Files
Modify the file paths in the `if __name__ == "__main__":` sections of each script:

```python
input_csv = "data/your_input_file.csv"
output_csv = "data/your_output_file.csv"
```

### Compare Different Binned Datasets
In `compare_model_data.py`, uncomment different lines to compare various binning methods:

```python
# binned_df = pd.read_csv("data/discretized_raw_sampled_500.csv")        # Equal-width
# binned_df = pd.read_csv("data/discretized_month_raw_sampled_500.csv")  # Month-based
binned_df = pd.read_csv("data/discretized_d3_raw_sampled_500.csv")     # Decision tree
```

## 📚 Key Concepts

### Binning/Discretization
The process of converting continuous variables into categorical ones by creating intervals or "bins". This can:
- Reduce noise in data
- Handle outliers better
- Create interpretable feature relationships
- Sometimes improve model performance

### Binning Methods Implemented:
1. **Equal-Width**: Simple, uniform intervals
2. **Month-Based**: Domain-specific seasonal grouping
3. **Supervised**: Target-aware optimal splitting using decision trees

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing techniques
- Feature engineering through binning
- Model comparison methodologies
- Impact of discretization on predictive performance
- Handling of missing values and categorical encoding
