import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from contracts.feature_contract import FeatureContract


NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch"]
CATEGORICAL_MAP  = {
    "sex":      {"male": 0, "female": 1},
    "embarked": {"S": 0, "C": 1, "Q": 2},
    "pclass":   {1: 0, 2: 1, 3: 2},
}
# Final feature order (must be stable — model input_dim depends on this)
FEATURE_ORDER = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]


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
        self._age_median  = float(df["age"].median())
        self._fare_median = float(df["fare"].median())

        # Fill with computed medians before fitting scaler
        df["age"]  = df["age"].fillna(self._age_median)
        df["fare"] = df["fare"].fillna(self._fare_median)

        numeric_block = df[NUMERIC_FEATURES].astype(float)
        self._scaler.fit(numeric_block)
        self._fitted = True
        return self

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sex"]      = df["sex"].str.lower().str.strip()
        df["embarked"] = df["embarked"].str.upper().str.strip().fillna("S")
        return df

    # ------------------------------------------------------------------
    # Inference path (single record dict → 1-D numpy array)
    # ------------------------------------------------------------------

    def transform(self, record: dict) -> np.ndarray:
        """
        Accepts one raw dict matching InputSchema.
        Returns shape (1, n_features) — ready for model.forward().
        """
        age      = float(record.get("age")  or self._age_median)
        fare     = float(record.get("fare") or self._fare_median)
        sibsp    = int(record.get("sibsp", 0))
        parch    = int(record.get("parch", 0))

        sex      = str(record.get("sex", "male")).lower().strip()
        embarked = str(record.get("embarked", "S")).upper().strip()
        pclass   = int(record.get("pclass", 3))

        # Encode categoricals
        sex_enc      = CATEGORICAL_MAP["sex"].get(sex, 0)
        embarked_enc = CATEGORICAL_MAP["embarked"].get(embarked, 0)
        pclass_enc   = CATEGORICAL_MAP["pclass"].get(pclass, 2)

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