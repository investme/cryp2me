"""
src/data/features.py
────────────────────
Computes all 18 input features from raw OHLCV data.
Each feature is normalised to be scale-invariant across tokens.

Feature list (matches FeatureConfig.all_features):
  Price (4):    open_pct, high_pct, low_pct, close_pct
  Volume (2):   volume_norm, volume_ratio
  EMAs (3):     ema10_dist, ema20_dist, ema34_dist
  Momentum (4): rsi14, macd, macd_signal, macd_hist
  Volatility(2):atr14_norm, bb_width
  Trend (3):    adx14, plus_di, minus_di
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — returns array same length as input (NaN-padded)."""
    k   = 2.0 / (period + 1)
    out = np.full(len(series), np.nan)
    if len(series) < period:
        return out
    out[period - 1] = series[:period].mean()
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    out    = np.full(len(close), np.nan)
    deltas = np.diff(close)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return out

    ag = gains[:period].mean()
    al = losses[:period].mean()
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr  = np.maximum(high - low,
          np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    out = np.full(len(tr), np.nan)
    out[period - 1] = tr[:period].mean()
    k = 1.0 / period
    for i in range(period, len(tr)):
        out[i] = tr[i] * k + out[i - 1] * (1 - k)
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (adx, plus_di, minus_di), all 0-100 scaled."""
    n = len(close)
    out_adx = np.full(n, np.nan)
    out_pdi = np.full(n, np.nan)
    out_mdi = np.full(n, np.nan)

    prev_h = np.roll(high,  1); prev_h[0] = high[0]
    prev_l = np.roll(low,   1); prev_l[0] = low[0]
    prev_c = np.roll(close, 1); prev_c[0] = close[0]

    up_move   = high  - prev_h
    down_move = prev_l - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))

    # Smooth with Wilder's MA
    atr14  = np.full(n, np.nan)
    pdm14  = np.full(n, np.nan)
    mdm14  = np.full(n, np.nan)

    atr14[period] = tr[1:period+1].sum()
    pdm14[period] = plus_dm[1:period+1].sum()
    mdm14[period] = minus_dm[1:period+1].sum()

    for i in range(period + 1, n):
        atr14[i] = atr14[i-1] - atr14[i-1]/period + tr[i]
        pdm14[i] = pdm14[i-1] - pdm14[i-1]/period + plus_dm[i]
        mdm14[i] = mdm14[i-1] - mdm14[i-1]/period + minus_dm[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi  = np.where(atr14 > 0, 100 * pdm14 / atr14, 0.0)
        mdi  = np.where(atr14 > 0, 100 * mdm14 / atr14, 0.0)
        dx   = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    out_pdi[period:] = pdi[period:]
    out_mdi[period:] = mdi[period:]

    # ADX = smoothed DX
    adx_smooth = np.full(n, np.nan)
    start = 2 * period
    if n > start:
        adx_smooth[start] = dx[period:start+1].mean()
        for i in range(start + 1, n):
            adx_smooth[i] = (adx_smooth[i-1] * (period - 1) + dx[i]) / period
    out_adx = adx_smooth

    return out_adx, out_pdi, out_mdi


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [time, open, high, low, close, volume]
    Output: DataFrame with all 18 normalised features + time column.
    Rows with NaN (warm-up period) are dropped.
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    t = df["time"].values

    # ── Price features: % change from previous close ──────────────────────────
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    open_pct  = (o - prev_c) / prev_c
    high_pct  = (h - prev_c) / prev_c
    low_pct   = (l - prev_c) / prev_c
    close_pct = (c - prev_c) / prev_c

    # ── Volume features ───────────────────────────────────────────────────────
    v_ma30 = _ema(v, 30)
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_norm  = np.log1p(v)
        volume_ratio = np.where(v_ma30 > 0, v / v_ma30, 1.0)

    # ── EMA distances (% from close) ──────────────────────────────────────────
    ema10 = _ema(c, 10)
    ema20 = _ema(c, 20)
    ema34 = _ema(c, 34)
    with np.errstate(divide='ignore', invalid='ignore'):
        ema10_dist = (c - ema10) / c
        ema20_dist = (c - ema20) / c
        ema34_dist = (c - ema34) / c

    # ── Momentum ──────────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14) / 100.0  # normalise 0-1

    # MACD
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line  = ema12 - ema26
    k9 = 2.0 / (9 + 1)
    macd_sig   = np.full(len(c), np.nan)
    # seed signal from first valid MACD value
    first_valid = np.where(~np.isnan(macd_line))[0]
    if len(first_valid) >= 9:
        seed_idx = first_valid[8]
        macd_sig[seed_idx] = np.nanmean(macd_line[first_valid[0]:seed_idx+1])
        for i in range(seed_idx + 1, len(c)):
            if not np.isnan(macd_line[i]):
                macd_sig[i] = macd_line[i] * k9 + macd_sig[i-1] * (1 - k9)

    macd_hist = macd_line - macd_sig
    # Normalise MACD by close price
    with np.errstate(divide='ignore', invalid='ignore'):
        macd_norm  = macd_line  / c
        msig_norm  = macd_sig   / c
        mhist_norm = macd_hist  / c

    # ── Volatility ────────────────────────────────────────────────────────────
    atr14 = _atr(h, l, c, 14)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr14_norm = atr14 / c

    # Bollinger Band width (normalised)
    bb_period = 20
    bb_mid    = _ema(c, bb_period)
    roll_std  = pd.Series(c).rolling(bb_period).std().values
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = np.where(bb_mid > 0, 2 * roll_std / bb_mid, 0.0)

    # ── Trend ─────────────────────────────────────────────────────────────────
    adx14, plus_di, minus_di = _adx(h, l, c, 14)
    adx14_norm    = adx14    / 100.0
    plus_di_norm  = plus_di  / 100.0
    minus_di_norm = minus_di / 100.0

    # ── Labels (targets) ─────────────────────────────────────────────────────
   # Future returns: % change from current close to T+1, T+2, T+3
    ret1 = (np.roll(c, -1) - c) / c
    ret2 = (np.roll(c, -2) - c) / c
    ret3 = (np.roll(c, -3) - c) / c
    # Invalidate look-ahead rows
    ret1[-1]  = np.nan
    ret2[-2:] = np.nan
    ret3[-3:] = np.nan
    # Direction labels: 1 = up, 0 = down
    dir1 = (ret1 > 0).astype(float)
    dir2 = (ret2 > 0).astype(float)
    dir3 = (ret3 > 0).astype(float)

    feat = pd.DataFrame({
        "time": t,
        # features
        "open_pct":    open_pct,
        "high_pct":    high_pct,
        "low_pct":     low_pct,
        "close_pct":   close_pct,
        "volume_norm": volume_norm,
        "volume_ratio":volume_ratio,
        "ema10_dist":  ema10_dist,
        "ema20_dist":  ema20_dist,
        "ema34_dist":  ema34_dist,
        "rsi14":       rsi14,
        "macd":        macd_norm,
        "macd_signal": msig_norm,
        "macd_hist":   mhist_norm,
        "atr14_norm":  atr14_norm,
        "bb_width":    bb_width,
        "adx14":       adx14_norm,
        "plus_di":     plus_di_norm,
        "minus_di":    minus_di_norm,
        # targets
        "ret1": ret1, "ret2": ret2, "ret3": ret3,
        "dir1": dir1, "dir2": dir2, "dir3": dir3,
        # raw close for prediction output
        "close": c,
    })

    # Drop warm-up NaN rows (first ~34 rows)
    feat = feat.dropna(subset=[f for f in feat.columns if f not in ["time", "close"]])
    feat = feat.reset_index(drop=True)
    # ── Per-feature standardisation ───────────────────────────────────────────
    # Brings all features to mean≈0, std≈1 so no single feature dominates
    # gradients. Fit on the full series (walk-forward CV handles leakage
    # at the fold level by fitting scaler on train split only).
    feature_cols = [
        "open_pct", "high_pct", "low_pct", "close_pct",
        "volume_norm", "volume_ratio",
        "ema10_dist", "ema20_dist", "ema34_dist",
        "rsi14", "macd", "macd_signal", "macd_hist",
        "atr14_norm", "bb_width",
        "adx14", "plus_di", "minus_di",
    ]
    for col in feature_cols:
        if col in feat.columns:
            mu  = feat[col].mean()
            sig = feat[col].std()
            if sig > 1e-8:
                feat[col] = (feat[col] - mu) / sig
    return feat


def build_all(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build features for all tickers."""
    from rich.console import Console
    from rich.progress import track
    console = Console()
    result = {}
    console.print("\n[bold cyan]⚙  Building features[/bold cyan]")
    for ticker, df in track(raw_data.items(), description="Engineering features..."):
        feat = build_features(df)
        if len(feat) >= 100:
            result[ticker] = feat
    console.print(f"[bold green]✓  Features ready for {len(result)} tickers[/bold green]\n")
    return result
