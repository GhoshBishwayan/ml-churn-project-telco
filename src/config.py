from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_PATH: Path = PROJECT_ROOT / "data" / "telco_churn.csv"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    WEEK1_REPORT_PATH: Path = REPORTS_DIR / "week1_findings.md"

    # Dataset settings (Telco churn default)
    TARGET_COL: str = "Churn"
    ID_COLS: tuple[str, ...] = ("customerID",)

    # Common telco numeric col that is often stored as text
    TOTAL_CHARGES_COL: str = "TotalCharges"
    TENURE_COL: str = "tenure"
    MONTHLY_CHARGES_COL: str = "MonthlyCharges"
    CONTRACT_COL: str = "Contract"
    PAYMENT_METHOD_COL: str = "PaymentMethod"
    INTERNET_SERVICE_COL: str = "InternetService"

CFG = Config()
