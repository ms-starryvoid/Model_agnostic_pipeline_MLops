import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from contracts.feature_contract import FeatureContract


NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CATEGORICAL_MAP  = {
    "Sex":      {"male": 0, "female": 1},
    "Embarked": {"S": 0, "C": 1, "Q": 2},
    "Pclass":   {1: 0, 2: 1, 3: 2},
}
# Final feature order (must be stable — model input_dim depends on this)
FEATURE_ORDER = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


class TitanicFeatureProcessor(FeatureContract):
    """
    Fit once on the training DataFrame.
    Transform one dict at a time at inference.
    Handles missing values the same way in both paths.
    """

    # Dataset-level fill values — set during fit, used during transform
    _age_median:  float = 28.0   # fallback before fit
    _fare_median: float = 14.45

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TitanicFeatureProcessor":
        df = self._clean_df(df)
        self._age_median  = float(df["Age"].median())
        self._fare_median = float(df["Fare"].median())

        # Fill with computed medians before fitting scaler
        df["Age"]  = df["Age"].fillna(self._age_median)
        df["Fare"] = df["Fare"].fillna(self._fare_median)

        numeric_block = df[NUMERIC_FEATURES].astype(float)
        self._scaler.fit(numeric_block)
        self._fitted = True
        return self

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Sex"]      = df["Sex"].str.lower().str.strip()
        df["Embarked"] = df["Embarked"].str.upper().str.strip().fillna("S")
        return df

    # ------------------------------------------------------------------
    # Inference path (single record dict → 1-D numpy array)
    # ------------------------------------------------------------------

    def transform(self, record: dict) -> np.ndarray:
        """
        Accepts one raw dict matching InputSchema.
        Returns shape (1, n_features) — ready for model.forward().
        """
        age = float(self._age_median if pd.isna(record.get("Age")) else record.get("Age"))
        fare = float(self._fare_median if pd.isna(record.get("Fare")) else record.get("Fare"))
        sibsp    = int(record.get("SibSp", 0))
        parch    = int(record.get("Parch", 0))

        sex      = str(record.get("Sex", "male")).lower().strip()
        embarked = str(record.get("Embarked", "S")).upper().strip()
        pclass   = int(record.get("Pclass", 3))

        # Encode categoricals
        sex_enc      = CATEGORICAL_MAP["Sex"].get(sex, 0)
        embarked_enc = CATEGORICAL_MAP["Embarked"].get(embarked, 0)
        pclass_enc   = CATEGORICAL_MAP["Pclass"].get(pclass, 2)

        # Scale numerics using fitted scaler
        numeric_block = np.array([[age, fare, sibsp, parch]], dtype=float)
        if self._fitted:
            numeric_block = self._scaler.transform(numeric_block)
        age_s, fare_s, sibsp_s, parch_s = numeric_block[0]

        # Assemble in FEATURE_ORDER: pclass, sex, age, sibsp, parch, fare, embarked
        features = np.array(
            [pclass_enc, sex_enc, age_s, sibsp_s, parch_s, fare_s, embarked_enc],
            dtype=np.float32,
        )
        return features.reshape(1, -1)   # (1, 7)

    def get_feature_names(self) -> list[str]:
        return FEATURE_ORDER