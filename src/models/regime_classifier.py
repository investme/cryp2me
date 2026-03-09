"""
src/models/regime_classifier.py
─────────────────────────────────
Hidden Markov Model (HMM) based market regime classifier.

Identifies 4 market regimes per hourly candle:
  0 — Quiet/Ranging    (low vol, no trend)
  1 — Trending Up      (low vol, uptrend)
  2 — Trending Down    (low vol, downtrend)
  3 — Volatile/Crisis  (high vol, any direction)

The regime label is added as a feature to the model, allowing the
LSTM/Transformer to condition its predictions on market state —
exactly like conditioning pier embedment solutions on soil type.

Physics analogy:
  Regime = environmental field state
  Price model = particle behaviour within that field
  Training per regime = solving PDEs with regime-specific boundary conditions

Usage:
    from src.models.regime_classifier import RegimeClassifier
    rc = RegimeClassifier()
    rc.fit(features_df)
    labels = rc.predict(features_df)   # array of 0/1/2/3
    features_df['regime'] = labels
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Optional


class RegimeClassifier:
    """
    4-state market regime classifier using Gaussian HMM.
    Falls back to rule-based classification if hmmlearn not installed.

    Input features used for regime detection:
      - volatility (ATR / close, rolling)
      - trend strength (ADX)
      - return sign (positive/negative momentum)
    """

    def __init__(
        self,
        n_states:    int   = 4,
        lookback:    int   = 24,   # hours for rolling stats
        cache_path:  Optional[Path] = None,
    ):
        self.n_states   = n_states
        self.lookback   = lookback
        self.cache_path = cache_path or Path("models/regime_classifier.pkl")
        self.model      = None
        self._fitted    = False

    # ── HMM features ─────────────────────────────────────────────────────────
    def _build_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build the 3 features used for regime detection.
        These must exist in df (they're part of our standard feature set).
        """
        features = []

        # 1. Normalised volatility (ATR / close or bb_width)
        if "atr14_norm" in df.columns:
            vol = df["atr14_norm"].values
        elif "bb_width" in df.columns:
            vol = df["bb_width"].values
        else:
            # Compute rolling std of returns as fallback
            if "close_pct" in df.columns:
                vol = pd.Series(df["close_pct"].values).rolling(self.lookback).std().fillna(0).values
            else:
                vol = np.zeros(len(df))
        features.append(vol)

        # 2. Trend strength (ADX normalised 0-1)
        if "adx14" in df.columns:
            adx = df["adx14"].values
        else:
            adx = np.zeros(len(df))
        features.append(adx)

        # 3. Return momentum (rolling mean of close_pct)
        if "close_pct" in df.columns:
            momentum = pd.Series(df["close_pct"].values).rolling(self.lookback).mean().fillna(0).values
        else:
            momentum = np.zeros(len(df))
        features.append(momentum)

        X = np.column_stack(features)
        # Replace any NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ── Rule-based fallback ───────────────────────────────────────────────────
    def _rule_based_regime(self, X: np.ndarray) -> np.ndarray:
        """
        Deterministic regime assignment when hmmlearn unavailable.
        Uses volatility + trend + momentum thresholds.

        Regime logic (like soil classification by measurable properties):
          High vol (>75th pct)                       → 3 (Volatile)
          Low vol + high ADX + positive momentum      → 1 (Trending Up)
          Low vol + high ADX + negative momentum      → 2 (Trending Down)
          Otherwise                                   → 0 (Ranging)
        """
        vol       = X[:, 0]
        adx       = X[:, 1]
        momentum  = X[:, 2]

        vol_hi    = np.percentile(vol[vol > 0], 75) if (vol > 0).any() else 0.05
        adx_hi    = 0.25   # ADX > 25 = trending

        labels = np.zeros(len(X), dtype=int)

        # Volatile regime
        labels[vol > vol_hi] = 3

        # Trending regimes (only where not already volatile)
        trending_mask = (labels == 0) & (adx > adx_hi)
        labels[trending_mask & (momentum > 0)] = 1
        labels[trending_mask & (momentum < 0)] = 2

        return labels

    def fit(self, df: pd.DataFrame) -> "RegimeClassifier":
        """Fit HMM on regime features extracted from df."""
        X = self._build_regime_features(df)

        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                verbose=False,
            )
            model.fit(X)
            self.model   = model
            self._fitted = True
            print(f"  ✓  HMM regime classifier fitted ({self.n_states} states)")
        except ImportError:
            print("  ℹ  hmmlearn not installed — using rule-based regime classifier")
            print("     Install with: pip install hmmlearn")
            self._fitted = True   # rule-based needs no fitting

        # Save
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({"model": self.model, "fitted": self._fitted}, f)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime labels (0-3) for each row in df."""
        X = self._build_regime_features(df)

        if self.model is not None:
            try:
                labels = self.model.predict(X)
                # Remap HMM states to interpretable regimes
                labels = self._remap_states(X, labels)
                return labels
            except Exception:
                pass

        return self._rule_based_regime(X)

    def _remap_states(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        HMM states are arbitrary integers — remap to interpretable regime IDs
        based on the mean volatility and momentum of each state.
        """
        vol      = X[:, 0]
        momentum = X[:, 2]
        n        = self.n_states

        state_vol  = np.array([vol[labels == s].mean()      if (labels == s).any() else 0 for s in range(n)])
        state_mom  = np.array([momentum[labels == s].mean() if (labels == s).any() else 0 for s in range(n)])

        # Highest vol → regime 3
        # Lowest vol + positive mom → regime 1
        # Lowest vol + negative mom → regime 2
        # Rest → regime 0

        remapped = np.zeros_like(labels)
        vol_rank = np.argsort(state_vol)

        remapped[labels == vol_rank[-1]] = 3   # highest vol = volatile
        for s in vol_rank[:-1]:
            if state_mom[s] > 0.0001:
                remapped[labels == s] = 1
            elif state_mom[s] < -0.0001:
                remapped[labels == s] = 2
            # else stays 0 (ranging)

        return remapped

    @classmethod
    def load(cls, cache_path: Path) -> "RegimeClassifier":
        rc = cls(cache_path=cache_path)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            rc.model   = data.get("model")
            rc._fitted = data.get("fitted", False)
        return rc

    def regime_name(self, label: int) -> str:
        return {0: "Ranging", 1: "Trending Up",
                2: "Trending Down", 3: "Volatile"}[label]
