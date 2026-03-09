"""
app/services/inference.py  (add to cryp2me-backend)
──────────────────────────────────────────────────────
ONNX Runtime inference service.
Loads all three ONNX models on startup, runs fast CPU inference per request.

Drop this file into cryp2me-backend/app/services/
and update the predict router to use it.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Feature columns must match training order exactly
FEATURE_COLS = [
    "open_pct", "high_pct", "low_pct", "close_pct",
    "volume_norm", "volume_ratio",
    "ema10_dist", "ema20_dist", "ema34_dist",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "atr14_norm", "bb_width",
    "adx14", "plus_di", "minus_di",
]

SEQ_LEN    = 60
N_FEATURES = 18


class ONNXInferenceService:
    """
    Loads LSTM + Transformer + MetaLearner ONNX models.
    Provides predict() method for FastAPI route.

    Usage:
        service = ONNXInferenceService("models/onnx")
        result  = service.predict(feature_df, current_price)
    """

    def __init__(self, onnx_dir: str = "models/onnx"):
        self.onnx_dir  = Path(onnx_dir)
        self._lstm:    Optional[ort.InferenceSession] = None
        self._tf:      Optional[ort.InferenceSession] = None
        self._meta:    Optional[ort.InferenceSession] = None
        self._loaded   = False

    def load(self) -> bool:
        """
        Load ONNX sessions. Returns True if successful.
        Called once at FastAPI startup.
        """
        lstm_path = self.onnx_dir / "lstm.onnx"
        tf_path   = self.onnx_dir / "transformer.onnx"
        meta_path = self.onnx_dir / "meta_learner.onnx"

        missing = [p for p in [lstm_path, tf_path, meta_path] if not p.exists()]
        if missing:
            logger.warning(
                f"ONNX models not found: {missing}. "
                "Predictions will return LOW_CONFIDENCE until models are trained (Phase 2)."
            )
            return False

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4

        self._lstm = ort.InferenceSession(str(lstm_path),  sess_options=opts)
        self._tf   = ort.InferenceSession(str(tf_path),    sess_options=opts)
        self._meta = ort.InferenceSession(str(meta_path),  sess_options=opts)
        self._loaded = True
        logger.info("✓ ONNX inference models loaded")
        return True

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def predict(
        self,
        feature_rows: List[dict],   # list of dicts with FEATURE_COLS keys
        current_price: float,
        confidence_threshold: float = 0.65,
    ) -> dict:
        """
        Runs ensemble inference on a sequence of feature rows.

        Args:
            feature_rows: last SEQ_LEN rows of computed features
            current_price: last known close price
            confidence_threshold: below this → LOW_CONFIDENCE

        Returns dict with signal, confidence, horizons
        """
        if not self._loaded:
            return self._stub_response(current_price)

        # Build input tensor
        x = self._build_input(feature_rows)  # (1, 60, 18)

        # Run LSTM
        lstm_reg, lstm_cls = self._lstm.run(None, {"input": x})

        # Run Transformer
        _, tf_cls = self._tf.run(None, {"input": x})

        # Run MetaLearner
        ensemble_cls = self._meta.run(
            None,
            {"lstm_probs": lstm_cls, "transformer_probs": tf_cls}
        )[0]  # (1, 3)

        probs = ensemble_cls[0]          # (3,) — P(up) for T+1, T+2, T+3
        rets  = lstm_reg[0]              # (3,) — % return predictions

        # Signal from T+1d
        p_up   = float(probs[0])
        p_down = 1.0 - p_up
        conf   = max(p_up, p_down)

        if conf < confidence_threshold:
            signal = "LOW_CONFIDENCE"
        elif p_up > p_down:
            signal = "BULLISH"
        else:
            signal = "BEARISH"

        horizons = []
        for i, (label, horizon_days) in enumerate(
            [("T+24h", 1), ("T+48h", 2), ("T+72h", 3)]
        ):
            pred_price = current_price * (1.0 + float(rets[i]))
            change_pct = float(rets[i]) * 100
            h_conf     = max(float(probs[i]), 1.0 - float(probs[i])) * 100
            horizons.append({
                "label":          label,
                "price":          round(pred_price, 6),
                "change_percent": round(change_pct, 4),
                "confidence":     round(h_conf, 2),
            })

        return {
            "signal":      signal,
            "confidence":  round(conf * 100, 2),
            "horizons":    horizons,
            "model_version": "v1.0-ensemble-onnx",
        }

    def _build_input(self, feature_rows: List[dict]) -> np.ndarray:
        """Convert feature dicts to model input array (1, SEQ_LEN, N_FEATURES)."""
        rows = feature_rows[-SEQ_LEN:]
        arr  = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
        offset = SEQ_LEN - len(rows)
        for i, row in enumerate(rows):
            for j, col in enumerate(FEATURE_COLS):
                val = row.get(col, 0.0)
                arr[0, offset + i, j] = float(val) if val is not None else 0.0
        arr = np.clip(arr, -10, 10)
        return arr

    def _stub_response(self, current_price: float) -> dict:
        """Returned when models are not yet trained."""
        return {
            "signal":     "LOW_CONFIDENCE",
            "confidence": 0.0,
            "horizons": [
                {"label": "T+24h", "price": current_price, "change_percent": 0.0, "confidence": 0.0},
                {"label": "T+48h", "price": current_price, "change_percent": 0.0, "confidence": 0.0},
                {"label": "T+72h", "price": current_price, "change_percent": 0.0, "confidence": 0.0},
            ],
            "model_version": "v0.0-phase1-stub",
        }


# ── Singleton (imported by FastAPI app) ───────────────────────────────────────
inference_service = ONNXInferenceService(onnx_dir="models/onnx")
