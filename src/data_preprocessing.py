"""
Data-preprocessing utilities for insurance cost prediction.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class InsuranceDataPreprocessor:
    """Preprocessor for the insurance dataset."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self, filepath: str) -> pd.DataFrame | None:
        """Load the insurance dataset."""
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None

    def basic_info(self, df: pd.DataFrame) -> None:
        """Display basic dataset information."""
        print("=== Dataset Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str = "charges",
        test_size: float = 0.3,
        random_state: int = 42,
    ):
        """Split into train/validation/test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 70 % train, 15 % val, 15 % test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=random_state
        )

        print(f"Train: {X_train.shape[0]}")
        print(f"Validation: {X_val.shape[0]}")
        print(f"Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_insurance_data() -> pd.DataFrame:
    """Quick helper for notebooks."""
    return pd.read_csv("data/raw/insurance.csv")