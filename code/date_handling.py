import pandas as pd
import os

def process_merged_data(input_file, output_no_weeks, output_months_only):
    """
    Process merged_raw_sampled_500.csv to:
    1. Remove week columns and output the result
    2. Keep weeks but transform dates to only months and output that
    
    Args:
        input_file (str): Path to the input CSV file
        output_no_weeks (str): Path to save the data without week columns
        output_months_only (str): Path to save the data with dates transformed to months
    """
    
    # Read the input data
    print("Reading merged_raw_sampled_500.csv...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Date column to datetime for processing
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Remove week columns (DayOfWeek) and output
    print("\n1. Processing data without week columns...")
    df_no_weeks = df.drop(columns=['DayOfWeek'])
    
    print(f"Data without weeks shape: {df_no_weeks.shape}")
    print(f"Columns after removing weeks: {list(df_no_weeks.columns)}")
    
    # Save the data without weeks
    df_no_weeks.to_csv(output_no_weeks, index=False)
    print(f"Saved data without weeks to: {output_no_weeks}")
    
    # 2. Keep weeks but transform dates to only months
    print("\n2. Processing data with dates transformed to months only...")
    df_months = df.copy()
    
    # Extract month from Date and replace the Date column, then rename to Month
    df_months['Date'] = df_months['Date'].dt.month
    df_months = df_months.rename(columns={'Date': 'Month'})
    
    print(f"Data with months only shape: {df_months.shape}")
    print(f"Columns: {list(df_months.columns)}")
    print(f"Month column now contains months (1-12): {sorted(df_months['Month'].unique())}")
    
    # Save the data with months only
    df_months.to_csv(output_months_only, index=False)
    print(f"Saved data with months only to: {output_months_only}")
    
    # Display sample of both outputs
    print("\n--- Sample of data without weeks ---")
    print(df_no_weeks.head())
    
    print("\n--- Sample of data with months only ---")
    print(df_months.head())
    
    return df_no_weeks, df_months

if __name__ == "__main__":
    # Change to the script directory to handle relative paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Define paths
    input_file = '../data/merged_raw_sampled_500.csv'
    output_no_weeks = '../data/merged_raw_sampled_500_date_only.csv'
    output_months_only = '../data/merged_raw_discretized_500_week_month.csv'
    
    process_merged_data(input_file, output_no_weeks, output_months_only)