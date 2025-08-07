import pandas as pd
import numpy as np

def sample_csv(input_file, output_file, n, random_state=42):
    df = pd.read_csv(input_file)
    sampled_df = df.sample(n=n, random_state=random_state)
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled {n} rows from {input_file} and saved to {output_file}")

def main():
    input_file = "data/merged_raw_data_5000.csv"
    output_file = "data/merged_raw_sampled_500.csv"
    n = 5000
    sample_csv(input_file, output_file, n)

if __name__ == "__main__":
    main()
