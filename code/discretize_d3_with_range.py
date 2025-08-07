import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def supervised_binning(feature, target, max_bins=10, min_samples_split=50):
    mask = ~(pd.isna(feature) | pd.isna(target))
    feature_clean = feature[mask]
    target_clean = target[mask]
    if len(feature_clean) == 0:
        return feature, []
    dt = DecisionTreeRegressor(
        max_leaf_nodes=max_bins,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(feature_clean.values.reshape(-1, 1), target_clean)
    tree = dt.tree_
    split_points = []
    def extract_splits(node_id):
        if tree.children_left[node_id] != tree.children_right[node_id]:
            split_points.append(tree.threshold[node_id])
            extract_splits(tree.children_left[node_id])
            extract_splits(tree.children_right[node_id])
    try:
        extract_splits(0)
        split_points = sorted(list(set(split_points)))
        bins = [-np.inf] + split_points + [np.inf]
    except:
        try:
            _, bins = pd.qcut(feature_clean, q=min(max_bins, len(feature_clean.unique())), 
                            retbins=True, duplicates='drop')
        except:
            bins = np.linspace(feature_clean.min(), feature_clean.max(), max_bins+1)
    binned_feature = pd.cut(feature, bins=bins, duplicates='drop')
    return binned_feature, bins

def discretize_and_save(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)
    df_clean_customers = df[~pd.isna(df['Customers']) & ~pd.isna(df['Sales'])]
    customers_binned, customers_bins = supervised_binning(df['Customers'], df['Sales'])
    customers_labels = []
    for i in range(len(customers_bins)-1):
        left = customers_bins[i] if customers_bins[i] != -np.inf else df['Customers'].min()
        right = customers_bins[i+1] if customers_bins[i+1] != np.inf else df['Customers'].max()
        customers_labels.append(f"({int(left)},{int(right)}]")
    df['Customers_binned'] = pd.cut(df['Customers'], bins=customers_bins, labels=customers_labels, duplicates='drop')
    df_clean_comp = df[~pd.isna(df['CompetitionDistance']) & ~pd.isna(df['Sales'])]
    comp_binned, comp_bins = supervised_binning(df['CompetitionDistance'], df['Sales'])
    comp_labels = []
    for i in range(len(comp_bins)-1):
        left = comp_bins[i] if comp_bins[i] != -np.inf else df['CompetitionDistance'].min()
        right = comp_bins[i+1] if comp_bins[i+1] != np.inf else df['CompetitionDistance'].max()
        comp_labels.append(f"({int(left)},{int(right)}]")
    df['CompetitionDistance_binned'] = pd.cut(df['CompetitionDistance'], bins=comp_bins, labels=comp_labels, duplicates='drop')
    df.drop(['Customers', 'CompetitionDistance'], axis=1, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Discretized CSV saved as '{output_csv}'")

def main(input_csv, output_csv):
    discretize_and_save(input_csv, output_csv)

if __name__ == "__main__":
    input_csv = "data/merged_raw_sampled_500.csv"
    output_csv = "data/discretized_d3_raw_sampled_500.csv"
    main(input_csv, output_csv)


