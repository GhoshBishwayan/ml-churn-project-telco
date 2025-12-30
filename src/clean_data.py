import pandas as pd

def clean_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_cols: tuple[str, ...] = ("customerID",),
    total_charges_col: str = "TotalCharges",
) -> pd.DataFrame:
    """
    Cleans common issues in churn datasets (especially Telco churn).
    - Removes duplicate rows
    - Strips whitespace in column names and object values
    - Converts TotalCharges to numeric
    - Ensures target column exists and values are normalized
    """
    df = df.copy()

    # Basic normalization
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates()

    # Strip whitespace in object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Drop ID columns if present
    for c in id_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Convert TotalCharges (often stored as " " or string)
    if total_charges_col in df.columns:
        df[total_charges_col] = pd.to_numeric(df[total_charges_col], errors="coerce")

    # Validate target
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    # Normalize churn labels (common: Yes/No)
    # Keep as string "Yes"/"No" for Week1 EDA (later we'll map to 0/1)
    df[target_col] = df[target_col].replace(
        {
            "yes": "Yes",
            "no": "No",
            "1": "Yes",
            "0": "No",
            "True": "Yes",
            "False": "No",
        }
    )

    # Drop rows where target missing/invalid
    df = df[df[target_col].isin(["Yes", "No"])].copy()

    return df


def data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Returns a structured summary useful for reports.
    """
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    summary = {
        "shape": df.shape,
        "num_columns": df.shape[1],
        "num_rows": df.shape[0],
        "missing_columns": missing.to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return summary
