"""
app/routers/predict.py  (UPDATED — replaces Phase 1 stub)
──────────────────────────────────────────────────────────
Phase 2: Real ONNX inference.

Copy this file to cryp2me-backend/app/routers/predict.py
and copy inference_service.py to cryp2me-backend/app/services/inference.py
"""

from fastapi import APIRouter, Query, HTTPException
from app.models import PredictionResponse
from app.services import fetch_candles
from app.services.inference import inference_service
from src.data.features import build_features
import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(
    ticker:   str,
    interval: str = Query("1d"),
):
    ticker = ticker.upper().strip()

    try:
        # Fetch enough candles for feature warm-up + sequence
        candles = await fetch_candles(ticker, interval)

        if len(candles) < 80:
            raise HTTPException(422, f"Insufficient data for {ticker}")

        # Build features
        df = pd.DataFrame([c.dict() for c in candles])
        feat_df = build_features(df)

        if len(feat_df) < 60:
            raise HTTPException(422, "Insufficient features after warm-up")

        # Get last 60 rows as feature dicts
        feature_rows  = feat_df.tail(60).to_dict(orient="records")
        current_price = float(candles[-1].close)

        # Run inference
        result = inference_service.predict(feature_rows, current_price)

        return PredictionResponse(
            ticker=ticker,
            signal=result["signal"],
            confidence=result["confidence"],
            horizons=result["horizons"],
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_version=result["model_version"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error for {ticker}: {e}")
        raise HTTPException(500, "Prediction failed")
