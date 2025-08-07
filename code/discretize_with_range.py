import pandas as pd
import numpy as np

def equal_width_binning(feature, num_bins=10):
    feature_clean = feature.dropna()
    if len(feature_clean) == 0:
        return feature, []
    min_val = feature_clean.min()
    max_val = feature_clean.max()
    bins = np.linspace(min_val, max_val, num_bins + 1)
    binned_feature = pd.cut(feature, bins=bins, duplicates='drop')
    return binned_feature, bins

def discretize_and_save(input_csv, output_csv, num_bins=10):
    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)
    customers_binned, customers_bins = equal_width_binning(df['Customers'], num_bins)
    customers_labels = []
    for i in range(len(customers_bins)-1):
        left = customers_bins[i]
        right = customers_bins[i+1]
        customers_labels.append(f"({int(left)},{int(right)}]")
    df['Customers_binned'] = pd.cut(df['Customers'], bins=customers_bins, labels=customers_labels, duplicates='drop')
    comp_binned, comp_bins = equal_width_binning(df['CompetitionDistance'], num_bins)
    comp_labels = []
    for i in range(len(comp_bins)-1):
        left = comp_bins[i]
        right = comp_bins[i+1]
        comp_labels.append(f"({int(left)},{int(right)}]")
    df['CompetitionDistance_binned'] = pd.cut(df['CompetitionDistance'], bins=comp_bins, labels=comp_labels, duplicates='drop')
    df.drop(['Customers', 'CompetitionDistance'], axis=1, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Discretized CSV saved as '{output_csv}'")

def main(input_csv, output_csv, num_bins=10):
    discretize_and_save(input_csv, output_csv, num_bins)

if __name__ == "__main__":
    input_csv = "data/merged_raw_sampled_500.csv"
    output_csv = "data/discretized_raw_sampled_500.csv"
    main(input_csv, output_csv, num_bins=10)
