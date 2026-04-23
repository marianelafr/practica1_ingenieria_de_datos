from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


@dataclass
class Practica1Preprocessing(BaseEstimator, TransformerMixin):

    low_null_threshold: float = 0.10
    high_null_threshold: float = 0.50
    low_cardinality_columns: List[str] = field(
        default_factory=lambda: [
            "home_ownership",
            "verification_status",
            "application_type",
            "purpose",
        ]
    )
    frequency_columns: List[str] = field(
        default_factory=lambda: ["emp_title", "addr_state"]
    )
    ordinal_columns: List[str] = field(default_factory=lambda: ["grade", "sub_grade"])
    binned_columns: List[str] = field(
        default_factory=lambda: ["ingresos_bins", "patrimonio_bins", "dti_bins"]
    )
    labels_4_tramos: List[str] = field(
        default_factory=lambda: ["Bajo", "Medio-Bajo", "Medio-Alto", "Alto"]
    )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        df = X.copy()

        self.input_columns_ = df.columns.tolist()

        nulls_perc = df.isnull().mean()
        self.columns_to_drop_ = nulls_perc[nulls_perc >= self.high_null_threshold].index.tolist()
        df = df.drop(columns=self.columns_to_drop_, errors="ignore")

        self.low_null_columns_ = nulls_perc[nulls_perc < self.low_null_threshold].index.tolist()
        self.mid_null_columns_ = nulls_perc[
            (nulls_perc >= self.low_null_threshold) & (nulls_perc < self.high_null_threshold)
        ].index.tolist()

        self.mode_values_: Dict[str, object] = {}
        for column in self.low_null_columns_:
            if column in df.columns and df[column].notna().any():
                self.mode_values_[column] = df[column].mode(dropna=True).iloc[0]

        self.median_values_: Dict[str, float] = {}
        for column in self.mid_null_columns_:
            if column in df.columns:
                numeric_column = pd.to_numeric(df[column], errors="coerce")
                if numeric_column.notna().any():
                    self.median_values_[column] = float(numeric_column.median())

        df = self._apply_null_handling(df)
        df = self._apply_basic_cleaning(df)

        available_ordinal = [c for c in self.ordinal_columns if c in df.columns]
        self.ordinal_columns_ = available_ordinal
        if self.ordinal_columns_:
            self.ordinal_encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.ordinal_encoder_.fit(df[self.ordinal_columns_].astype(str))

        self.low_cardinality_columns_ = [c for c in self.low_cardinality_columns if c in df.columns]
        if self.low_cardinality_columns_:
            self.ohe_categorical_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.ohe_categorical_.fit(df[self.low_cardinality_columns_].astype(str))

        self.frequency_columns_ = [c for c in self.frequency_columns if c in df.columns]
        self.frequency_maps_: Dict[str, Dict[object, float]] = {}
        for column in self.frequency_columns_:
            self.frequency_maps_[column] = df[column].value_counts(normalize=True, dropna=False).to_dict()

        df = self._apply_encoders(df)
        df = self._add_engineered_features(df, fit=True)

        self.binned_columns_ = [c for c in self.binned_columns if c in df.columns]
        self.ohe_bins_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe_bins_.fit(df[self.binned_columns_].astype(str))

        df = self._encode_binned_columns(df)

        self.numeric_columns_ = df.select_dtypes(include="number").columns.tolist()
        self.scaler_ = RobustScaler()
        self.scaler_.fit(df[self.numeric_columns_])

        self.output_columns_ = df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        df = X.copy()
        missing_input_columns = [c for c in self.input_columns_ if c not in df.columns]
        for column in missing_input_columns:
            df[column] = np.nan

        df = df[self.input_columns_]
        df = df.drop(columns=self.columns_to_drop_, errors="ignore")
        df = self._apply_null_handling(df)
        df = self._apply_basic_cleaning(df)
        df = self._apply_encoders(df)
        df = self._add_engineered_features(df, fit=False)
        df = self._encode_binned_columns(df)

        numeric_columns = [c for c in self.numeric_columns_ if c in df.columns]
        df[numeric_columns] = self.scaler_.transform(df[numeric_columns])

        df = df.reindex(columns=self.output_columns_, fill_value=0)
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _apply_null_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for column, mode_value in self.mode_values_.items():
            if column in df.columns:
                df[column] = df[column].fillna(mode_value)

        for column, median_value in self.median_values_.items():
            if column in df.columns:
                numeric_column = pd.to_numeric(df[column], errors="coerce")
                df[column] = numeric_column.fillna(median_value)

        return df

    def _apply_basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "zip_code" in df.columns:
            df = df.drop(columns="zip_code")

        if "earliest_cr_line" in df.columns:
            dates = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
            df["earliest_cr_line_year"] = dates.dt.year
            df["earliest_cr_line_month"] = dates.dt.month
            df = df.drop(columns="earliest_cr_line")

        if "term" in df.columns:
            df["term"] = (
                df["term"]
                .astype(str)
                .str.replace(" months", "", regex=False)
                .replace("nan", np.nan)
            )
            df["term"] = pd.to_numeric(df["term"], errors="coerce")

        if "emp_length" in df.columns:
            mapped = df["emp_length"].replace({"10+ years": "11", "< 1 year": "0"})
            extracted = mapped.astype(str).str.extract(r"(\d+)")[0]
            df["emp_length"] = pd.to_numeric(extracted, errors="coerce")

        return df

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if getattr(self, "ordinal_columns_", None):
            encoded = self.ordinal_encoder_.transform(df[self.ordinal_columns_].astype(str))
            df[self.ordinal_columns_] = encoded.astype(int)

        if getattr(self, "low_cardinality_columns_", None):
            transformed = self.ohe_categorical_.transform(df[self.low_cardinality_columns_].astype(str))
            transformed_df = pd.DataFrame(
                transformed,
                columns=self.ohe_categorical_.get_feature_names_out(self.low_cardinality_columns_),
                index=df.index,
            )
            df = pd.concat([df.drop(columns=self.low_cardinality_columns_), transformed_df], axis=1)

        for column in getattr(self, "frequency_columns_", []):
            mapping = self.frequency_maps_[column]
            df[column] = df[column].map(mapping).fillna(0.0)

        return df

    def _add_engineered_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()

        if {"fico_range_low", "fico_range_high"}.issubset(df.columns):
            df["prob_low"] = (df["fico_range_low"] - 300) / (850 - 300)
            df["prob_high"] = (df["fico_range_high"] - 300) / (850 - 300)
            df["prob_medio"] = (df["prob_low"] + df["prob_high"]) / 2
            df["prediction"] = np.where(df["prob_medio"] > 0.67, 0, 1)
            df["fico_medio"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round()

        if {"dti", "annual_inc"}.issubset(df.columns):
            df["deuda_total"] = df["dti"] * df["annual_inc"]

        if fit:
            self.annual_inc_bins_ = self._learn_qcut_bins(df["annual_inc"], q=4)
            self.tot_cur_bal_bins_ = self._learn_qcut_bins(df["tot_cur_bal"], q=4)

        if "annual_inc" in df.columns:
            df["ingresos_bins"] = pd.cut(
                df["annual_inc"],
                bins=self.annual_inc_bins_,
                labels=self.labels_4_tramos[: len(self.annual_inc_bins_) - 1],
                include_lowest=True,
            )

        if "tot_cur_bal" in df.columns:
            df["patrimonio_bins"] = pd.cut(
                df["tot_cur_bal"],
                bins=self.tot_cur_bal_bins_,
                labels=self.labels_4_tramos[: len(self.tot_cur_bal_bins_) - 1],
                include_lowest=True,
            )

        if "dti" in df.columns:
            df["dti_bins"] = pd.cut(
                df["dti"],
                bins=[-np.inf, 20, 35, 50, np.inf],
                labels=["Saludable", "Aceptable", "Alerta", "Riesgo Alto"],
            )

        return df

    def _encode_binned_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        transformed = self.ohe_bins_.transform(df[self.binned_columns_].astype(str))
        transformed_df = pd.DataFrame(
            transformed,
            columns=self.ohe_bins_.get_feature_names_out(self.binned_columns_),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=self.binned_columns_), transformed_df], axis=1)
        return df

    @staticmethod
    def _learn_qcut_bins(series: pd.Series, q: int) -> np.ndarray:
        clean_series = pd.to_numeric(series, errors="coerce").dropna()
        _, bins = pd.qcut(clean_series, q=q, retbins=True, duplicates="drop")
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _check_is_fitted(self):
        required_attrs = [
            "input_columns_",
            "columns_to_drop_",
            "mode_values_",
            "median_values_",
            "binned_columns_",
            "ohe_bins_",
            "scaler_",
            "output_columns_",
        ]
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise AttributeError(
                "Practica1Preprocessing no esta entrenada. Ejecuta fit o fit_transform antes de transform."
            )
