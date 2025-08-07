import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

def preprocess(df):
    df = df.copy()
    df = df.dropna(subset=['Sales'])
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def evaluate_model(df, dataset_name):
    df = preprocess(df)
    X = df.drop(columns=['Sales'])
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {dataset_name.upper()} DATASET ===")
    print("R² Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    coef = pd.Series(np.abs(model.coef_), index=X.columns)
    top_features = coef.sort_values(ascending=False).head(10)
    print("Top 10 Features by Coefficient Magnitude:\n", top_features)

if __name__ == "__main__":
    original_df = pd.read_csv("data/merged_raw_sampled_500.csv")
    # binned_df = pd.read_csv("data/discretized_raw_sampled_500.csv")
    # binned_df = pd.read_csv("data/discretized_month_raw_sampled_500.csv")
    binned_df = pd.read_csv("data/discretized_d3_raw_sampled_500.csv")
    evaluate_model(original_df, "Original")
    evaluate_model(binned_df, "Binned")
