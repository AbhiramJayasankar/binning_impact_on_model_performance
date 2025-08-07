import pandas as pd

def bin_months(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)
    month_bins = [0, 3, 6, 9, 12]
    month_labels = ["(0,3]", "(3,6]", "(6,9]", "(9,12]"]
    df['Month_binned'] = pd.cut(df['Month'], bins=month_bins, labels=month_labels, include_lowest=True)
    df.drop(['Month'], axis=1, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Discretized CSV with month binning saved as '{output_path}'")

if __name__ == "__main__":
    input_path = "data/merged_raw_sampled_500.csv"
    output_path = "data/discretized_month_raw_sampled_500.csv"
    bin_months(input_path, output_path)
