# cryp2me-ml

Machine Learning engine for cryp2me.ai — LSTM + Transformer ensemble for crypto price prediction.

## Architecture

```
Input: (batch, 60 days, 18 features)
         │
    ┌────┴────┐
    │  LSTM   │  3-layer stacked, 256→128→64 hidden
    └────┬────┘
         │  direction probabilities (3 horizons)
    ┌────┴────────────┐
    │ Meta-Learner    │  lightweight NN trained on OOF predictions
    └────┬────────────┘
    ┌────┴────────────┐
    │  Transformer    │  4-layer, d_model=128, 8 heads, causal mask
    └────┬────────────┘
         │
    Ensemble output: signal + confidence + predicted prices
```

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Quick test run (5 tickers, 10 epochs)
python scripts/train.py --fast

# Full production training
python scripts/train.py

# Jupyter notebook (step-by-step walkthrough)
jupyter notebook notebooks/phase2_training.ipynb
```

## 18 Input Features

| Category    | Features |
|-------------|----------|
| Price (4)   | open_pct, high_pct, low_pct, close_pct |
| Volume (2)  | volume_norm, volume_ratio |
| EMAs (3)    | ema10_dist, ema20_dist, ema34_dist |
| Momentum (4)| rsi14, macd, macd_signal, macd_hist |
| Volatility (2)| atr14_norm, bb_width |
| Trend (3)   | adx14, plus_di, minus_di |

## Accuracy Strategy (84% target)

| Method | Expected Accuracy |
|--------|------------------|
| Random baseline | 50% |
| LSTM (price only) | 55–62% |
| LSTM (18 features) | 63–70% |
| Ensemble (LSTM + Transformer) | 70–78% |
| + Confidence thresholding (65%+) | 78–84%+ |

Key insight: **confidence thresholding** — only output a signal when
`max(P_up, P_down) ≥ 0.65`. Below this threshold the model returns
`LOW_CONFIDENCE` rather than a wrong prediction. This trades coverage
(fewer signals) for precision (higher accuracy on the signals given).

## After Training

Copy outputs to the backend to activate real predictions:

```bash
cp models/onnx/*.onnx ../cryp2me-backend/models/onnx/
cp src/export/inference_service.py ../cryp2me-backend/app/services/
cp src/export/predict_router_v2.py ../cryp2me-backend/app/routers/predict.py
```

## Folder Structure

```
configs/
  config.py           # All hyperparameters in one place
src/
  data/
    collector.py      # Binance OHLCV download + caching
    features.py       # 18-feature engineering pipeline
    dataset.py        # PyTorch Dataset + walk-forward splits
  models/
    lstm_model.py     # Stacked LSTM with dual output heads
    transformer_model.py  # Temporal Transformer with causal masking
    ensemble.py       # MetaLearner + EnsemblePredictor
  training/
    trainer.py        # Training loop, loss, early stopping, LR schedule
    walk_forward_cv.py # Walk-forward cross-validation
  evaluation/
    metrics.py        # Accuracy report + threshold analysis
  export/
    onnx_export.py    # PyTorch → ONNX export + verification
    inference_service.py  # Production ONNX inference (for backend)
    predict_router_v2.py  # Updated FastAPI router (replace Phase 1 stub)
scripts/
  train.py            # Master orchestration script
notebooks/
  phase2_training.ipynb  # Step-by-step walkthrough
```
# cryp2me
# cryp2me
# cryp2me
# crp2me
# crp2me
# cryp2me
