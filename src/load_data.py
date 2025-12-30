from pathlib import Path
import pandas as pd

def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Check the CSV file.")
    return df
