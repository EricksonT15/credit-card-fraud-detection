"""
preprocessing.py

Reusable data cleaning and preprocessing functions for the
Credit Card Fraud Detection capstone project.

These functions are extracted from notebooks/01_data_exploration.ipynb
and notebooks/02_preprocessing.ipynb so that cleaning logic is defined
once and can be imported consistently across all notebooks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def load_raw_data(filepath: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Load the raw ULB Credit Card Fraud dataset.

    Parameters
    ----------
    filepath : str
        Path to the creditcard.csv file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataset.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {filepath}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values across all columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table of missing count and percentage per column.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct.round(4)
    })
    n_missing_cols = (summary["Missing Count"] > 0).sum()
    if n_missing_cols == 0:
        print(f"No missing values found across all {df.shape[1]} columns.")
    else:
        print(f"WARNING: {n_missing_cols} columns contain missing values.")
    return summary


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Deduplicated dataset with reset index.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    print(f"Removed {n_removed:,} duplicate rows "
          f"({n_before:,} -> {len(df_clean):,})")
    return df_clean


def scale_time_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply RobustScaler to the Time and Amount columns, replacing
    them with Time_scaled and Amount_scaled. RobustScaler is used
    because Amount is heavily right-skewed with extreme outliers
    that represent legitimate high-value transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Time' and 'Amount' columns.

    Returns
    -------
    pd.DataFrame
        Dataset with Time/Amount replaced by their scaled versions.
    """
    df = df.copy()
    robust = RobustScaler()
    df["Time_scaled"] = robust.fit_transform(df[["Time"]])
    df["Amount_scaled"] = robust.fit_transform(df[["Amount"]])
    df = df.drop(columns=["Time", "Amount"])
    print("Applied RobustScaler to Time and Amount "
          "(replaced with Time_scaled, Amount_scaled)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-derived engineered features:
    - Amount_log: log1p transform of Amount_scaled (compresses skew)
    - High_amount: binary flag for transactions above the 75th percentile

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Amount_scaled' column (run scale_time_amount first).

    Returns
    -------
    pd.DataFrame
        Dataset with Amount_log and High_amount columns added.
    """
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount_scaled"].clip(lower=0))

    q75 = df["Amount_scaled"].quantile(0.75)
    df["High_amount"] = (df["Amount_scaled"] > q75).astype(int)

    print("Engineered features added: Amount_log, High_amount "
          f"(High_amount threshold = Q75 = {q75:.4f})")
    return df


def clean_pipeline(filepath: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Run the full cleaning pipeline end to end:
    load -> check missing -> remove duplicates -> scale -> engineer.

    Parameters
    ----------
    filepath : str
        Path to the raw creditcard.csv file.

    Returns
    -------
    pd.DataFrame
        Fully cleaned and feature-engineered dataset, ready for
        feature selection and modelling.
    """
    df = load_raw_data(filepath)
    check_missing_values(df)
    df = remove_duplicates(df)
    df = scale_time_amount(df)
    df = engineer_features(df)
    print(f"\nClean pipeline complete: {df.shape[0]:,} rows x "
          f"{df.shape[1]} columns")
    return df
