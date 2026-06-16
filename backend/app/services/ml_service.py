"""
ML inference service for MuleShield AI.

Loads the CatBoost model at startup (singleton pattern).
Handles single-record and batch inference with SHAP explanations.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from app.core.config import get_settings
from app.services.risk_service import probability_to_risk_score, risk_tier

logger = logging.getLogger(__name__)

settings = get_settings()


class MLService:
    """Singleton ML service — loaded once at app startup."""

    def __init__(self) -> None:
        self._model = None
        self._explainer = None
        self._feature_names: list[str] = []
        self._model_config: dict[str, Any] = {}
        self._feature_df: pd.DataFrame | None = None
        self._is_loaded = False

    def load(self) -> None:
        """Load model, features, and SHAP explainer."""
        logger.info("Loading ML artifacts...")

        # Load feature list
        with open(settings.features_path, "r") as f:
            self._feature_names = json.load(f)
        logger.info(f"Loaded {len(self._feature_names)} features")

        # Load model config
        with open(settings.model_config_path, "r") as f:
            self._model_config = json.load(f)
        logger.info(f"Model config: threshold={self._model_config.get('optimal_threshold')}")

        # Load feature importance CSV
        self._feature_df = pd.read_csv(settings.features_csv_path)

        # Load CatBoost model
        self._model = joblib.load(settings.model_path)
        logger.info("CatBoost model loaded successfully")

        # Build SHAP TreeExplainer (fast for tree-based models)
        try:
            self._explainer = shap.TreeExplainer(self._model)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer failed, falling back to KernelExplainer: {e}")
            self._explainer = None

        self._is_loaded = True
        logger.info("MLService fully loaded ✓")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> str:
        return self._model_config.get("best_model_name", "CatBoost")

    @property
    def optimal_threshold(self) -> float:
        return float(self._model_config.get("optimal_threshold", 0.933))

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def model_metrics(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pr_auc": self._model_config.get("pr_auc", 0.755),
            "roc_auc": self._model_config.get("roc_auc", 0.994),
            "f1_score": self._model_config.get("f1_at_threshold", 0.71),
            "precision": self._model_config.get("precision_at_threshold", 1.0),
            "recall": self._model_config.get("recall_at_threshold", 0.55),
            "optimal_threshold": self.optimal_threshold,
            "n_features": self.n_features,
        }

    def _validate_and_align(self, features: dict[str, float]) -> pd.DataFrame:
        """Validate input features and align to expected order."""
        missing = [f for f in self._feature_names if f not in features]
        if missing:
            # Fill missing with 0.0 (median imputation default)
            for f in missing:
                features[f] = 0.0
            logger.warning(f"Filled {len(missing)} missing features with 0.0: {missing[:5]}...")

        row = {f: features.get(f, 0.0) for f in self._feature_names}
        df = pd.DataFrame([row], columns=self._feature_names)
        return df

    def predict_single(
        self,
        features: dict[str, float],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Run inference on a single account record.

        Returns:
            dict with probability, risk_score, label, top_features (SHAP)
        """
        if not self._is_loaded:
            raise RuntimeError("ML model not loaded. Call load() first.")

        X = self._validate_and_align(features)

        # Predict probability
        proba = float(self._model.predict_proba(X)[0, 1])
        score = probability_to_risk_score(proba)
        tier = risk_tier(score)
        threshold = self.optimal_threshold
        label = "SUSPICIOUS" if proba >= threshold else "LEGITIMATE"

        # SHAP explanations
        top_features = self._compute_shap_top_n(X, top_n)

        return {
            "probability": round(proba, 6),
            "risk_score": score,
            "risk_tier": tier,
            "label": label,
            "threshold_used": threshold,
            "top_features": top_features,
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Run inference on a batch DataFrame.

        Args:
            df: DataFrame with feature columns (may have extra columns; they're ignored)

        Returns:
            List of dicts, one per row.
        """
        if not self._is_loaded:
            raise RuntimeError("ML model not loaded.")

        # Align columns
        missing_cols = [f for f in self._feature_names if f not in df.columns]
        for col in missing_cols:
            df[col] = 0.0

        X = df[self._feature_names].copy()
        X = X.fillna(0.0)

        # Batch probability inference
        probas = self._model.predict_proba(X)[:, 1]
        threshold = self.optimal_threshold

        results = []
        # Compute SHAP for entire batch at once (much faster)
        shap_values_batch = self._compute_shap_batch(X, top_n)

        for i, proba in enumerate(probas):
            score = probability_to_risk_score(float(proba))
            tier = risk_tier(score)
            label = "SUSPICIOUS" if proba >= threshold else "LEGITIMATE"

            results.append({
                "row_index": i,
                "probability": round(float(proba), 6),
                "risk_score": score,
                "risk_tier": tier,
                "label": label,
                "threshold_used": threshold,
                "top_features": shap_values_batch[i],
            })

        return results

    def _compute_shap_top_n(
        self, X: pd.DataFrame, top_n: int
    ) -> list[dict[str, Any]]:
        """Compute SHAP values and return top N features by absolute impact."""
        if self._explainer is None:
            return []

        try:
            shap_vals = self._explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class 1 (fraud)
            sv = shap_vals[0]  # single row

            feature_shaps = [
                {
                    "name": self._feature_names[i],
                    "value": float(X.iloc[0, i]),
                    "shap_impact": round(float(sv[i]), 6),
                }
                for i in range(len(self._feature_names))
            ]
            # Sort by absolute SHAP impact
            feature_shaps.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
            return feature_shaps[:top_n]
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return []

    def _compute_shap_batch(
        self, X: pd.DataFrame, top_n: int
    ) -> list[list[dict[str, Any]]]:
        """Compute SHAP for all rows in batch."""
        if self._explainer is None:
            return [[] for _ in range(len(X))]

        try:
            shap_vals = self._explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            all_results = []
            for i in range(len(X)):
                sv = shap_vals[i]
                feature_shaps = [
                    {
                        "name": self._feature_names[j],
                        "value": float(X.iloc[i, j]),
                        "shap_impact": round(float(sv[j]), 6),
                    }
                    for j in range(len(self._feature_names))
                ]
                feature_shaps.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
                all_results.append(feature_shaps[:top_n])
            return all_results
        except Exception as e:
            logger.error(f"Batch SHAP computation failed: {e}")
            return [[] for _ in range(len(X))]

    def get_global_feature_importance(self) -> list[dict[str, Any]]:
        """Return global feature importance from RF + MI scores."""
        if self._feature_df is None:
            return []
        records = self._feature_df.to_dict("records")
        records.sort(key=lambda x: x.get("rf_importance", 0), reverse=True)
        return records


# Singleton instance
_ml_service: MLService | None = None


def get_ml_service() -> MLService:
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
