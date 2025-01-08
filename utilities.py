from typing import List

import pandas as pd

def find_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and counts missing values in a DataFrame, including zeroes, empty strings, and NaN values.

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame with counts of zeroes, empty strings, and NaN values for each column.
    """
    zeroes = (df == 0).sum()
    empty_strings = (df.replace(r"^\s*$", "", regex=True) == "").sum()
    nas = df.isna().sum()
    combined_counts = pd.DataFrame({
        "Zeroes": zeroes,
        "Empty Strings": empty_strings,
        "NaN": nas
        })
    return combined_counts

def find_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Detects outliers in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect outliers in.

    Returns:
        DataFrame containing the outliers for each feature.
    """
    outliers_list = []
    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame.")
            continue

        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        feature_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        if not feature_outliers.empty:
            print(f"Outliers in '{feature}':")
            print(feature_outliers[feature], end="\n\n")
            outliers_list.append(feature_outliers)
        else:
            print(f"No outliers in '{feature}'")

    if outliers_list:
        outliers = pd.concat(outliers_list)
        outliers = outliers[features]
    else:
        outliers = pd.DataFrame(columns=features)
        
    return outliers