from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures, ProbeFeatureSelection
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

@dataclass
class Practica1Filtering(BaseEstimator, TransformerMixin):

    corr_threshold: float = 0.9
    variance_threshold: float = 0.95
    probe_estimator: BaseEstimator = field(
        default_factory=lambda: RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
    )
    scoring: str = "roc_auc"
    n_probes: int = 10
    distribution: str = "normal"
    cv: int = 3
    random_state: int = 150
    confirm_variables: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("Practica1Filtering necesita `y` en fit por usar ProbeFeatureSelection.")

        df = X.copy()
        self.input_columns_ = df.columns.tolist()

        self.correlated_dropper_ = DropCorrelatedFeatures(
            threshold=self.corr_threshold,
            method="pearson",
            missing_values="ignore",
        )
        df = self.correlated_dropper_.fit_transform(df)
        self.after_correlation_columns_ = df.columns.tolist()

        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        variance_array = self.variance_selector_.fit_transform(df)
        self.variance_columns_ = self.variance_selector_.get_feature_names_out(df.columns)
        self.low_variance_dropped_columns_ = [
            column for column in self.after_correlation_columns_ if column not in list(self.variance_columns_)
        ]
        df = pd.DataFrame(variance_array, columns=self.variance_columns_, index=X.index)

        self.probe_selector_ = ProbeFeatureSelection(
            estimator=clone(self.probe_estimator),
            variables=None,
            scoring=self.scoring,
            n_probes=self.n_probes,
            distribution=self.distribution,
            cv=self.cv,
            random_state=self.random_state,
            confirm_variables=self.confirm_variables,
        )
        df = self.probe_selector_.fit_transform(df, y)

        self.output_columns_ = df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        df = X.copy()
        missing_columns = [c for c in self.input_columns_ if c not in df.columns]
        for column in missing_columns:
            df[column] = 0

        df = df[self.input_columns_]
        df = self.correlated_dropper_.transform(df)

        variance_array = self.variance_selector_.transform(df)
        df = pd.DataFrame(variance_array, columns=self.variance_columns_, index=X.index)

        df = self.probe_selector_.transform(df)
        df = df.reindex(columns=self.output_columns_, fill_value=0)
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def correlated_features_to_drop_(self):
        self._check_is_fitted()
        return self.correlated_dropper_.features_to_drop_

    @property
    def low_variance_features_to_drop_(self):
        self._check_is_fitted()
        return self.low_variance_dropped_columns_

    @property
    def probe_features_to_drop_(self):
        self._check_is_fitted()
        return self.probe_selector_.features_to_drop_

    def _check_is_fitted(self):
        required_attrs = [
            "input_columns_",
            "correlated_dropper_",
            "after_correlation_columns_",
            "variance_selector_",
            "variance_columns_",
            "low_variance_dropped_columns_",
            "probe_selector_",
            "output_columns_",
        ]
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise AttributeError(
                "Practica1Filtering no esta entrenada. Ejecuta fit o fit_transform antes de transform."
            )
