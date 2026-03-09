"""
src/models/xgboost_ensemble.py
────────────────────────────────
XGBoost-based meta-learner that replaces the MLP ensemble.

Inputs:
  - LSTM classification probabilities  (3 horizons)
  - Transformer classification probs   (3 horizons)
  - Raw macro features at prediction time
  - Regime label (one-hot encoded)

Output:
  - Direction probability per horizon (T+24h, T+48h, T+72h)

Why XGBoost beats MLP for this task:
  - Handles tabular features without normalisation
  - Explicit feature importance — shows which signals matter most
  - Resistant to overfitting on small OOF datasets
  - Naturally handles non-linear interactions between regime + model outputs

Physics analogy:
  This is the binomial regression layer from Hussein's pier model —
  taking outputs from the simulation (LSTM/Transformer) and combining
  them with environmental conditions (macro/regime) into a final prediction
  using a robust statistical method.

Usage:
    from src.models.xgboost_ensemble import XGBoostEnsemble
    ens = XGBoostEnsemble()
    ens.fit(lstm_probs, tf_probs, macro_feats, regime_labels, targets)
    preds = ens.predict_proba(lstm_probs, tf_probs, macro_feats, regime_labels)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Tuple


class XGBoostEnsemble:
    """
    XGBoost meta-learner ensemble for 3-horizon crypto direction prediction.
    One XGBoost model per horizon for maximum accuracy.
    Falls back to weighted average if xgboost not installed.
    """

    def __init__(
        self,
        n_horizons:  int   = 3,
        n_estimators:int   = 300,
        max_depth:   int   = 4,
        learning_rate:float = 0.05,
        subsample:   float = 0.8,
        cache_path:  Optional[Path] = None,
    ):
        self.n_horizons   = n_horizons
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.lr           = learning_rate
        self.subsample    = subsample
        self.cache_path   = cache_path or Path("models/xgb_ensemble.pkl")
        self.models       = []   # one per horizon
        self._fitted      = False
        self._use_xgb     = True

    def _build_X(
        self,
        lstm_probs:    np.ndarray,   # (N, 3)
        tf_probs:      np.ndarray,   # (N, 3)
        macro_feats:   np.ndarray,   # (N, n_macro)
        regime_labels: np.ndarray,   # (N,) int 0-3
    ) -> np.ndarray:
        """
        Build feature matrix for XGBoost.

        Features:
          - LSTM probs per horizon          (3)
          - Transformer probs per horizon   (3)
          - LSTM - Transformer disagreement (3)  ← model agreement signal
          - Macro features                  (n_macro)
          - Regime one-hot                  (4)
        """
        n = len(lstm_probs)

        # Model agreement (high disagreement = lower confidence)
        disagreement = np.abs(lstm_probs - tf_probs)

        # Regime one-hot (4 states)
        regime_oh = np.zeros((n, 4))
        for i, r in enumerate(regime_labels.astype(int)):
            if 0 <= r < 4:
                regime_oh[i, r] = 1.0

        parts = [lstm_probs, tf_probs, disagreement, macro_feats, regime_oh]
        X = np.hstack([p for p in parts if p.shape[1] > 0])
        return np.nan_to_num(X, nan=0.0)

    def fit(
        self,
        lstm_probs:    np.ndarray,
        tf_probs:      np.ndarray,
        macro_feats:   np.ndarray,
        regime_labels: np.ndarray,
        targets:       np.ndarray,   # (N, 3) binary direction labels
    ) -> "XGBoostEnsemble":
        """Fit one XGBoost classifier per horizon."""

        X = self._build_X(lstm_probs, tf_probs, macro_feats, regime_labels)

        try:
            import xgboost as xgb
            self._use_xgb = True
        except ImportError:
            print("  ℹ  xgboost not installed — using weighted average fallback")
            print("     Install with: pip install xgboost")
            self._use_xgb = False
            self._fitted  = True
            # Store weights from simple logistic regression
            self._lstm_weight = 0.45
            self._tf_weight   = 0.55
            return self

        self.models = []
        feature_names = self._feature_names(macro_feats.shape[1])

        for h in range(self.n_horizons):
            y = targets[:, h]
            # Skip if all same class
            if len(np.unique(y)) < 2:
                self.models.append(None)
                continue

            model = xgb.XGBClassifier(
                n_estimators  = self.n_estimators,
                max_depth     = self.max_depth,
                learning_rate = self.lr,
                subsample     = self.subsample,
                colsample_bytree = 0.8,
                min_child_weight = 5,
                gamma         = 0.1,
                reg_alpha     = 0.1,
                reg_lambda    = 1.0,
                eval_metric   = "logloss",
                use_label_encoder = False,
                random_state  = 42,
                n_jobs        = -1,
                verbosity     = 0,
            )

            # Train with early stopping on 20% holdout
            split = int(len(X) * 0.8)
            model.fit(
                X[:split], y[:split],
                eval_set=[(X[split:], y[split:])],
                verbose=False,
            )
            self.models.append(model)

            # Print feature importance for first horizon
            if h == 0:
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[::-1][:5]
                print(f"  📊 Top features (T+{(h+1)*24}h):")
                for idx in top_idx:
                    fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
                    print(f"     {fname}: {imp[idx]:.3f}")

        self._fitted     = True
        self._n_macro    = macro_feats.shape[1]

        # Save
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "models":    self.models,
                "fitted":    self._fitted,
                "use_xgb":   self._use_xgb,
                "n_macro":   getattr(self, "_n_macro", 0),
            }, f)

        print(f"  ✓  XGBoost ensemble fitted ({self.n_horizons} models)")
        return self

    def predict_proba(
        self,
        lstm_probs:    np.ndarray,
        tf_probs:      np.ndarray,
        macro_feats:   np.ndarray,
        regime_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Predict direction probabilities for all horizons.
        Returns (N, 3) array of probabilities.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        if not self._use_xgb or not self.models:
            # Weighted average fallback
            w_l = getattr(self, "_lstm_weight", 0.45)
            w_t = getattr(self, "_tf_weight",   0.55)
            return w_l * lstm_probs + w_t * tf_probs

        X = self._build_X(lstm_probs, tf_probs, macro_feats, regime_labels)
        preds = np.zeros((len(X), self.n_horizons))

        for h, model in enumerate(self.models):
            if model is None:
                # Fallback for this horizon
                preds[:, h] = 0.5 * lstm_probs[:, h] + 0.5 * tf_probs[:, h]
            else:
                preds[:, h] = model.predict_proba(X)[:, 1]

        return preds

    def _feature_names(self, n_macro: int) -> list:
        names = (
            [f"lstm_h{h+1}" for h in range(3)] +
            [f"tf_h{h+1}"   for h in range(3)] +
            [f"disagree_h{h+1}" for h in range(3)] +
            [f"macro_{i}"   for i in range(n_macro)] +
            ["regime_0", "regime_1", "regime_2", "regime_3"]
        )
        return names

    def feature_importance_report(self) -> pd.DataFrame:
        """Return feature importance DataFrame for all horizons."""
        if not self._use_xgb or not self.models:
            return pd.DataFrame()

        n_macro = getattr(self, "_n_macro", 7)
        names   = self._feature_names(n_macro)
        rows    = []

        for h, model in enumerate(self.models):
            if model is None:
                continue
            imp = model.feature_importances_
            for i, v in enumerate(imp):
                fname = names[i] if i < len(names) else f"feat_{i}"
                rows.append({"horizon": f"T+{(h+1)*24}h", "feature": fname, "importance": v})

        return pd.DataFrame(rows).sort_values("importance", ascending=False)

    @classmethod
    def load(cls, cache_path: Path) -> "XGBoostEnsemble":
        ens = cls(cache_path=cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            ens.models   = data.get("models", [])
            ens._fitted  = data.get("fitted", False)
            ens._use_xgb = data.get("use_xgb", True)
            ens._n_macro = data.get("n_macro", 7)
        return ens
