import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def supervised_binning(feature, target, max_bins=10, min_samples_split=12):
    """
    Perform supervised binning using decision tree regressor.
    
    Parameters:
    feature: pandas Series - The feature to bin
    target: pandas Series - The target variable for supervised binning
    max_bins: int - Maximum number of bins
    min_samples_split: int - Minimum samples required to split
    
    Returns:
    binned_feature: pandas Series - The binned feature
    bins: list - The bin edges
    """
    # Remove NaN values
    mask = ~(pd.isna(feature) | pd.isna(target))
    feature_clean = feature[mask]
    target_clean = target[mask]
    
    if len(feature_clean) == 0:
        return feature, []
    
    # Fit decision tree
    dt = DecisionTreeRegressor(
        max_leaf_nodes=max_bins,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(feature_clean.values.reshape(-1, 1), target_clean)
    
    # Extract split points from the tree
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
        # Fallback to quantile-based binning
        try:
            _, bins = pd.qcut(feature_clean, q=min(max_bins, len(feature_clean.unique())), 
                            retbins=True, duplicates='drop')
        except:
            bins = np.linspace(feature_clean.min(), feature_clean.max(), max_bins+1)
    
    binned_feature = pd.cut(feature, bins=bins, duplicates='drop')
    return binned_feature, bins

def bin_competition_distance(input_csv, output_csv):
    """
    Read CSV file, apply decision tree binning to CompetitionDistance, and save the result.
    
    Parameters:
    input_csv: str - Path to input CSV file
    output_csv: str - Path to output CSV file
    """
    print(f"Loading data from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    print(f"Original data shape: {df.shape}")
    print(f"CompetitionDistance statistics:")
    print(df['CompetitionDistance'].describe())
    
    # Apply supervised binning to CompetitionDistance using Sales as target
    print("\nApplying decision tree binning to CompetitionDistance...")
    comp_binned, comp_bins = supervised_binning(df['CompetitionDistance'], df['Sales'])
    
    # Create meaningful labels for the bins
    comp_labels = []
    for i in range(len(comp_bins)-1):
        left = comp_bins[i] if comp_bins[i] != -np.inf else df['CompetitionDistance'].min()
        right = comp_bins[i+1] if comp_bins[i+1] != np.inf else df['CompetitionDistance'].max()
        comp_labels.append(f"({int(left)},{int(right)}]")
    
    # Replace the original CompetitionDistance with binned version
    df['CompetitionDistance_binned'] = pd.cut(df['CompetitionDistance'], bins=comp_bins, labels=comp_labels, duplicates='drop')
    
    # Drop the original CompetitionDistance column
    df.drop(['CompetitionDistance'], axis=1, inplace=True)
    
    print(f"\nBinned CompetitionDistance into {len(comp_bins)-1} bins:")
    print(f"Bin edges: {[f'{x:.1f}' if x != -np.inf and x != np.inf else str(x) for x in comp_bins]}")
    print(f"Bin labels: {comp_labels}")
    
    print(f"\nBinned CompetitionDistance value counts:")
    print(df['CompetitionDistance_binned'].value_counts().sort_index())
    
    # Save the result
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed data saved as '{output_csv}'")
    print(f"Final data shape: {df.shape}")

def main():
    input_csv = "data/merged_raw_discretized_500_week_month.csv"
    output_csv = "data/merged_raw_discretized_500_week_month_comp_binned.csv"
    bin_competition_distance(input_csv, output_csv)

if __name__ == "__main__":
    main()