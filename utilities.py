"""
Utility functions for data preprocessing, statistical analysis, and model evaluation.

This module provides various helper functions to assist in data exploration, 
model building, and evaluation, including functions for detecting missing values, 
identifying outliers, visualizing features and correlations, building linear and 
logistic regression models, and evaluating models through confusion matrices and 
residual plots.
"""

from typing import List

from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



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
    combined_counts = pd.DataFrame(
        {"Zeroes": zeroes, "Empty Strings": empty_strings, "NaN": nas}
    )
    return combined_counts


def find_outliers(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Detects outliers in multiple features using the IQR method.

    Args:
        df: DataFrame containing the data.
        features: List of features to detect outliers in.

    Returns:
        DataFrame containing the outliers for each feature and a DataFrame
        containing analysis for each feature (outlier count, percentage, IQR bounds,
        and flagged values).
    """
    outlier_reports = []
    outlier_indices = set()
    total_rows = len(df)

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_count = mask.sum()
        outlier_percentage = (outlier_count / total_rows) * 100

        flagged_values = "None"
        if outlier_count > 0:
            flagged_values = (
                f"[{df[feature][mask].min():.2f}, {df[feature][mask].max():.2f}]"
            )
            outlier_indices.update(df[mask].index)

        outlier_reports.append(
            {
                "Feature Name": feature,
                "Outliers": outlier_count,
                "Percentage": f"{outlier_percentage:.2f}%",
                "IQR Bounds": f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                "Flagged Values": flagged_values,
            }
        )

    outlier_reports_df = pd.DataFrame(outlier_reports)
    outlier_reports_df = outlier_reports_df[outlier_reports_df["Outliers"] > 0]

    if not outlier_reports_df.empty:
        display(Markdown("**Feature-wise Outlier Analysis**"))
        display(outlier_reports_df)
        display(Markdown("**All Outliers**"))
        outliers = df.loc[list(outlier_indices), features]
        display(outliers)
    else:
        print("**No features with outliers detected**")


def plot_features(df: pd.DataFrame, target: str) -> None:
    """
    Plots the distribution of each feature in the DataFrame and its relationship with the target variable.

    Args:
        df: The DataFrame containing the features and the target variable.
        target: The name of the target variable (column) in the DataFrame.

    Returns:
        None: Displays the plots but does not return any values.
    """
    features = [col for col in df.columns if col != target]
    rows = len(features)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 3))

    for i, feature in enumerate(features):
        sns.histplot(df[feature], kde=True, edgecolor="black", bins=15, ax=axes[i, 0])
        sns.scatterplot(x=df[feature], y=df[target], ax=axes[i, 1])

    fig.suptitle(f"Feature Distribution and Relationship with {target}", y=1)
    plt.tight_layout()
    plt.show()


def check_vif(df: pd.DataFrame, features: List[str]) -> None:
    """
    Calculates and prints the Variance Inflation Factor (VIF) for each feature
    in the dataset, excluding the specified features.

    Parameters:
        df: The input DataFrame containing the dataset.
        features: A list of column names to exclude from the VIF calculation.

    Returns:
        None
    """
    X = df.drop(columns=features)
    X_with_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(1, X_with_const.shape[1])
    ]
    vif_data["VIF"] = vif_data["VIF"].apply(lambda x: f"{x:.2f}")
    print(vif_data)


def split_dataset(df: pd.DataFrame, target: str, size: float = 0.2) -> list:
    """
    Splits the dataframe into training and testing sets.

    Args:
        df: The input data.
        target: The target variable.
        test_size: Proportion of the data to be used for testing.

    Returns:
        X_train (pd.DataFrame), X_test (pd.DataFrame), y_train (pd.Series), y_test (pd.Series)
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=size, random_state=42)


def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardizes the training and test data using StandardScaler.

    Args:
        X_train: The training features.
        X_test: The test features.

    Returns:
        Train and test feature DataFrames after standardization.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled


def build_ols_model(df: pd.DataFrame, features: List[str], target: str):
    """
    Builds and fits an OLS regression model.

    Args:
        df: The input DataFrame.
        features: The list of features to drop.
        target: The target variable.

    Returns:
        model: The fitted OLS model.
        X_test_scaled_const: The standardized test data with a constant added.
        y_test: The test target values.
    """
    X = df.drop(columns=features)

    X_train, X_test, y_train, y_test = split_dataset(X, target)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    X_train_scaled_const = add_constant(X_train_scaled)
    X_test_scaled_const = add_constant(X_test_scaled)

    model = OLS(y_train, X_train_scaled_const).fit()

    return (model, X_test_scaled_const, y_test)


def build_logistic_model(df: pd.DataFrame, target: str):
    """
    Trains a logistic regression model using scikit-learn.

    Args:
        X_train_scaled: The standardized training features.
        y_train: The target variable for the training set.

    Returns:
        model: The trained LogisticRegression model.
        X_test_scaled_const: The standardized test data.
        y_test: The test target values.
    """
    X_train, X_test, y_train, y_test = split_dataset(df, target)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)

    return (logreg, X_test_scaled, y_test)


def plot_confusion_matrix(
    y_test: pd.Series, y_pred: pd.Series, ticks, model_type: str
) -> None:
    """
    Plots a confusion matrix for the predicted and actual values using a heatmap.

    Parameters:
        y_test: The true values.
        y_pred: The predicted values.
        ticks: A list of tick values for the axes of the confusion matrix.
        model_type: The type of model being evaluated, included in the plot title.

    Returns:
        None
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=ticks, yticklabels=ticks)
    plt.xlabel(f"Predicted {y_test.name}")
    plt.ylabel(f"Actual {y_test.name}")
    plt.title(f"{model_type} model confusion matrix")
    plt.show()


def plot_corr_matrix(df: pd.DataFrame) -> None:
    """
    Plots a heatmap of the correlation matrix for the numerical features in the DataFrame.

    Parameters:
        df: The input DataFrame containing numerical features.

    Returns:
        None
    """
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, vmax=1, vmin=-1, cmap="vlag", annot=True)
    plt.title("Correlations heatmap")
    plt.show()


def plot_model_residuals(y_test: pd.Series, y_pred: pd.Series) -> None:
    """
    Plots the residuals and actual vs. predicted values for a regression model.

    Parameters:
        y_test: The true target values.
        y_pred: The predicted target values.

    Returns:
        None
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
    plt.xlabel(f"Actual {y_test.name}")
    plt.ylabel(f"Predicted {y_test.name}")
    plt.title(f"Actual vs. Predicted {y_test.name} values")

    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=15, edgecolor="black", alpha=0.7, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residuals Distribution")

    plt.subplot(1, 3, 3)
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()
    plt.show()
