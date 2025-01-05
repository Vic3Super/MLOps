import json

from google.cloud import bigquery

def extract_data(data):
    # Select only relevant columns
    df = data[["Trip Seconds", "Trip Miles", "Trip Total", "Company"]]

    # Remove outliers using IQR
    def remove_outliers(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    df = remove_outliers(df, ["Trip Seconds", "Trip Miles", "Trip Total"])

    return df

