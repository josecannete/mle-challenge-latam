import joblib
import pandas as pd

from typing import Tuple, Union

from datetime import datetime

from sklearn.linear_model import LogisticRegression


def get_minutes_diff(data: pd.DataFrame) -> int:
    """Calculate the difference in minutes between 'Fecha-O' and 'Fecha-I' columns.

    Args:
        data (pd.DataFrame): DataFrame with 'Fecha-O' and 'Fecha-I' columns.

    Returns:
        int: difference in minutes.
    """
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    minutes_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return minutes_diff


class DelayModel:
    _FEATURE_COLS: list[str] = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]
    _DELAY_THRESHOLD: int = 15

    def __init__(self):
        self._model = LogisticRegression(class_weight="balanced")

    def _create_delay_dataframe(
        self, data: pd.DataFrame, target_column_name: str
    ) -> pd.DataFrame:
        """Create a DataFrame with the target column.

        Args:
            data (pd.DataFrame): DataFrame with 'Fecha-O' and 'Fecha-I' columns.
            target_column_name (str): name of the target column.

        Returns:
            pd.DataFrame: DataFrame with the target column.
        """
        delay_df = pd.DataFrame()
        delay_df[target_column_name] = data.apply(lambda x: get_minutes_diff(x), axis=1)
        delay_df[target_column_name] = delay_df[target_column_name].apply(
            lambda x: 1 if x > self._DELAY_THRESHOLD else 0
        )
        return delay_df

    def save_model(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path (str): path to save the model.
        """
        joblib.dump(self._model, path)

    def load_model(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path (str): path to load the model.
        """
        self._model = joblib.load(path)

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        selected_features = pd.DataFrame(
            columns=self._FEATURE_COLS, dtype=int, index=data.index
        )

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        present_features = list(set(features.columns) & set(self._FEATURE_COLS))

        selected_features[present_features] = features[present_features]
        selected_features.fillna(0, inplace=True)

        if target_column:
            target = self._create_delay_dataframe(data, target_column)
            return selected_features, target

        return selected_features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> list[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (list[int]): predicted targets.
        """
        return self._model.predict(features).tolist()
