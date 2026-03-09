"""
src/data/features.py
────────────────────
Phase 4 — Complete integrated feature engineering.
35 input features total:

  Price    (4):  open_pct, high_pct, low_pct, close_pct
  Volume   (6):  volume_norm, volume_ratio, vol_divergence,
                 buy_sell_ratio, vol_momentum, obv_norm
  EMAs     (3):  ema10_dist, ema20_dist, ema34_dist
  Momentum (4):  rsi14, macd, macd_signal, macd_hist
  Volatility(2): atr14_norm, bb_width
  Trend    (3):  adx14, plus_di, minus_di
  Macro    (9):  dxy_pct, gold_pct, spx_pct, yield_10y_chg,
                 yield_curve, vix_level, vix_chg,
                 funding_rate, oi_change
  Regime   (4):  regime_0, regime_1, regime_2, regime_3
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


# ── Feature column registry ───────────────────────────────────────────────────

TECHNICAL_COLS = [
    "open_pct", "high_pct", "low_pct", "close_pct",
    "volume_norm", "volume_ratio", "vol_divergence",
    "buy_sell_ratio", "vol_momentum", "obv_norm",
    "ema10_dist", "ema20_dist", "ema34_dist",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "atr14_norm", "bb_width",
    "adx14", "plus_di", "minus_di",
]  # 22 technical features

MACRO_COLS = [
    "dxy_pct", "gold_pct", "spx_pct",
    "yield_10y_chg", "yield_curve",
    "vix_level", "vix_chg",
    "funding_rate", "oi_change",
]  # 9 macro features

REGIME_COLS = [
    "regime_0", "regime_1", "regime_2", "regime_3",
]  # 4 regime one-hot features

FEATURE_COLS = TECHNICAL_COLS + MACRO_COLS + REGIME_COLS  # 35 total

TARGET_REG_COLS = ["ret1", "ret2", "ret3"]
TARGET_CLS_COLS = ["dir1", "dir2", "dir3"]

N_FEATURES = len(FEATURE_COLS)  # 35


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average — returns array same length as input."""
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
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
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
    prev_h = np.roll(high,  1); prev_h[0] = high[0]
    prev_l = np.roll(low,   1); prev_l[0] = low[0]
    prev_c = np.roll(close, 1); prev_c[0] = close[0]

    up_move   = high  - prev_h
    down_move = prev_l - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr        = np.maximum(high - low,
                np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))

    atr14 = np.full(n, np.nan); pdm14 = np.full(n, np.nan); mdm14 = np.full(n, np.nan)
    atr14[period] = tr[1:period+1].sum()
    pdm14[period] = plus_dm[1:period+1].sum()
    mdm14[period] = minus_dm[1:period+1].sum()

    for i in range(period + 1, n):
        atr14[i] = atr14[i-1] - atr14[i-1]/period + tr[i]
        pdm14[i] = pdm14[i-1] - pdm14[i-1]/period + plus_dm[i]
        mdm14[i] = mdm14[i-1] - mdm14[i-1]/period + minus_dm[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr14 > 0, 100 * pdm14 / atr14, 0.0)
        mdi = np.where(atr14 > 0, 100 * mdm14 / atr14, 0.0)
        dx  = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    out_pdi = np.full(n, np.nan); out_mdi = np.full(n, np.nan)
    out_pdi[period:] = pdi[period:]; out_mdi[period:] = mdi[period:]

    adx_smooth = np.full(n, np.nan)
    start = 2 * period
    if n > start:
        adx_smooth[start] = dx[period:start+1].mean()
        for i in range(start + 1, n):
            adx_smooth[i] = (adx_smooth[i-1] * (period - 1) + dx[i]) / period

    return adx_smooth, out_pdi, out_mdi


# ── Volume helpers ────────────────────────────────────────────────────────────

def _compute_volume_features(
    c: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    v: np.ndarray,
    close_pct: np.ndarray,
    v_ma30: np.ndarray,
    volume_ratio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 4 advanced volume features.

    Returns: vol_divergence, buy_sell_ratio, vol_momentum, obv_norm
    """
    # 1. Volume divergence — price direction vs volume direction
    #    +1 = agreement (healthy trend), -1 = divergence (warning)
    #    High predictive value: divergence often precedes reversals
    vol_dir      = np.sign(np.diff(v, prepend=v[0]))
    price_dir    = np.sign(close_pct)
    vol_divergence = (price_dir * vol_dir).astype(float)

    # 2. Buy/sell pressure ratio (Tick Rule approximation)
    #    Estimates what fraction of volume was buying vs selling
    #    Based on where close landed within the high-low range
    hl_range     = np.where((h - l) > 1e-10, h - l, 1e-10)
    buy_vol      = v * ((c - l) / hl_range)
    sell_vol     = v - buy_vol
    with np.errstate(divide='ignore', invalid='ignore'):
        buy_sell_ratio = np.where(
            sell_vol > 1e-10,
            buy_vol / sell_vol,
            1.0
        )
    # Clip extreme ratios and log-scale for stability
    buy_sell_ratio = np.log1p(np.clip(buy_sell_ratio, 0, 10))

    # 3. Volume momentum — is volume accelerating or decelerating?
    #    Accelerating volume = trend strengthening
    #    Decelerating volume = trend weakening
    v_ma20      = _ema(volume_ratio, 20)
    vol_momentum = volume_ratio - np.where(np.isnan(v_ma20), 1.0, v_ma20)

    # 4. On-Balance Volume (OBV) normalised
    #    Cumulative volume flow — rising OBV = accumulation, falling = distribution
    #    One of the oldest and most reliable volume indicators
    obv         = np.cumsum(np.sign(close_pct) * v)
    obv_std     = pd.Series(obv).rolling(168).std().values   # 1-week rolling std
    obv_mean    = pd.Series(obv).rolling(168).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        obv_norm = np.where(
            obv_std > 1e-8,
            (obv - obv_mean) / obv_std,
            0.0
        )

    return vol_divergence, buy_sell_ratio, vol_momentum, obv_norm


# ── Regime classifier (lazy import) ──────────────────────────────────────────

def _get_regime_labels(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Fit and predict regime labels using the RegimeClassifier.
    Returns array of ints (0-3), length = len(feat_df).
    Falls back to zeros if regime classifier unavailable.
    """
    try:
        from src.models.regime_classifier import RegimeClassifier
        rc = RegimeClassifier()
        rc.fit(feat_df)
        return rc.predict(feat_df)
    except Exception:
        return np.zeros(len(feat_df), dtype=int)


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame] = None,
    regime_labels: Optional[np.ndarray] = None,
    fit_regime: bool = True,
) -> pd.DataFrame:
    """
    Build all 35 features from raw OHLCV + optional macro data.

    Args:
        df:            Raw OHLCV DataFrame [time, open, high, low, close, volume]
        macro_df:      Optional macro features aligned to df's timestamps
                       (output of macro_collector.add_macro_to_df)
        regime_labels: Optional pre-computed regime labels (0-3)
                       If None and fit_regime=True, will fit+predict internally
        fit_regime:    Whether to fit regime classifier if labels not provided

    Returns:
        DataFrame with 35 feature columns + target columns + time + close
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    t = df["time"].values

    # ── Price features ────────────────────────────────────────────────────────
    prev_c    = np.roll(c, 1); prev_c[0] = c[0]
    open_pct  = (o - prev_c) / prev_c
    high_pct  = (h - prev_c) / prev_c
    low_pct   = (l - prev_c) / prev_c
    close_pct = (c - prev_c) / prev_c

    # ── Volume features (existing) ────────────────────────────────────────────
    v_ma30 = _ema(v, 30)
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_norm  = np.log1p(v)
        volume_ratio = np.where(v_ma30 > 0, v / v_ma30, 1.0)

    # ── Volume features (new) ─────────────────────────────────────────────────
    vol_divergence, buy_sell_ratio, vol_momentum, obv_norm = \
        _compute_volume_features(c, h, l, v, close_pct, v_ma30, volume_ratio)

    # ── EMA distances ─────────────────────────────────────────────────────────
    ema10 = _ema(c, 10); ema20 = _ema(c, 20); ema34 = _ema(c, 34)
    with np.errstate(divide='ignore', invalid='ignore'):
        ema10_dist = (c - ema10) / c
        ema20_dist = (c - ema20) / c
        ema34_dist = (c - ema34) / c

    # ── Momentum ──────────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14) / 100.0

    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    k9        = 2.0 / (9 + 1)
    macd_sig  = np.full(len(c), np.nan)
    first_valid = np.where(~np.isnan(macd_line))[0]
    if len(first_valid) >= 9:
        seed_idx = first_valid[8]
        macd_sig[seed_idx] = np.nanmean(macd_line[first_valid[0]:seed_idx+1])
        for i in range(seed_idx + 1, len(c)):
            if not np.isnan(macd_line[i]):
                macd_sig[i] = macd_line[i] * k9 + macd_sig[i-1] * (1 - k9)
    macd_hist = macd_line - macd_sig
    with np.errstate(divide='ignore', invalid='ignore'):
        macd_norm  = macd_line / c
        msig_norm  = macd_sig  / c
        mhist_norm = macd_hist / c

    # ── Volatility ────────────────────────────────────────────────────────────
    atr14 = _atr(h, l, c, 14)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr14_norm = atr14 / c

    bb_mid   = _ema(c, 20)
    roll_std = pd.Series(c).rolling(20).std().values
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = np.where(bb_mid > 0, 2 * roll_std / bb_mid, 0.0)

    # ── Trend ─────────────────────────────────────────────────────────────────
    adx14, plus_di, minus_di = _adx(h, l, c, 14)
    adx14_norm    = adx14    / 100.0
    plus_di_norm  = plus_di  / 100.0
    minus_di_norm = minus_di / 100.0

    # ── Labels ────────────────────────────────────────────────────────────────
    ret1 = (np.roll(c, -1) - c) / c
    ret2 = (np.roll(c, -2) - c) / c
    ret3 = (np.roll(c, -3) - c) / c
    ret1[-1] = np.nan; ret2[-2:] = np.nan; ret3[-3:] = np.nan
    dir1 = (ret1 > 0).astype(float)
    dir2 = (ret2 > 0).astype(float)
    dir3 = (ret3 > 0).astype(float)

    # ── Assemble technical DataFrame ──────────────────────────────────────────
    feat = pd.DataFrame({
        "time":           t,
        "open_pct":       open_pct,
        "high_pct":       high_pct,
        "low_pct":        low_pct,
        "close_pct":      close_pct,
        "volume_norm":    volume_norm,
        "volume_ratio":   volume_ratio,
        "vol_divergence": vol_divergence,
        "buy_sell_ratio": buy_sell_ratio,
        "vol_momentum":   vol_momentum,
        "obv_norm":       obv_norm,
        "ema10_dist":     ema10_dist,
        "ema20_dist":     ema20_dist,
        "ema34_dist":     ema34_dist,
        "rsi14":          rsi14,
        "macd":           macd_norm,
        "macd_signal":    msig_norm,
        "macd_hist":      mhist_norm,
        "atr14_norm":     atr14_norm,
        "bb_width":       bb_width,
        "adx14":          adx14_norm,
        "plus_di":        plus_di_norm,
        "minus_di":       minus_di_norm,
        "ret1": ret1, "ret2": ret2, "ret3": ret3,
        "dir1": dir1, "dir2": dir2, "dir3": dir3,
        "close": c,
    })

    # Drop warm-up NaN rows
    feat = feat.dropna(subset=[f for f in feat.columns
                               if f not in ["time", "close"]])
    feat = feat.reset_index(drop=True)

    # ── Macro features ────────────────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        # Align macro to technical features by position
        for col in MACRO_COLS:
            if col in macro_df.columns:
                vals = macro_df[col].values
                if len(vals) >= len(feat):
                    feat[col] = vals[-len(feat):]
                else:
                    feat[col] = 0.0
            else:
                feat[col] = 0.0
    else:
        for col in MACRO_COLS:
            feat[col] = 0.0

    # ── Regime features ───────────────────────────────────────────────────────
    if regime_labels is not None:
        labels = regime_labels
    elif fit_regime:
        labels = _get_regime_labels(feat)
    else:
        labels = np.zeros(len(feat), dtype=int)

    # Align label length
    if len(labels) > len(feat):
        labels = labels[-len(feat):]
    elif len(labels) < len(feat):
        labels = np.pad(labels, (len(feat) - len(labels), 0), constant_values=0)

    for r in range(4):
        feat[f"regime_{r}"] = (labels == r).astype(float)

    # ── Standardise all features ──────────────────────────────────────────────
    for col in FEATURE_COLS:
        if col in feat.columns:
            mu  = feat[col].mean()
            sig = feat[col].std()
            if sig > 1e-8:
                feat[col] = (feat[col] - mu) / sig

    return feat


def build_all(
    raw_data:  Dict[str, pd.DataFrame],
    macro_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build features for all tickers.

    Args:
        raw_data:  {ticker: ohlcv_df}
        macro_map: Optional {ticker: macro_df} — pre-aligned macro per ticker
    """
    from rich.console import Console
    from rich.progress import track
    console = Console()

    result = {}
    console.print("\n[bold cyan]⚙  Building features[/bold cyan]")

    # Fit one regime classifier on all data combined for consistency
    regime_clf = None
    try:
        from src.models.regime_classifier import RegimeClassifier
        # Build tech features first pass (no macro, no regime) for fitting
        console.print("  🎯 Fitting regime classifier on all tickers...")
        tech_frames = []
        for df in raw_data.values():
            tmp = build_features(df, fit_regime=False)
            tech_frames.append(tmp)
        all_tech = pd.concat(tech_frames, ignore_index=True)
        regime_clf = RegimeClassifier(cache_path=__import__("pathlib").Path("models/regime_classifier.pkl"))
        regime_clf.fit(all_tech)
        console.print("  ✓  Regime classifier fitted")
    except Exception as e:
        console.print(f"  ⚠  Regime classifier unavailable: {e}")

    for ticker, df in track(raw_data.items(), description="Engineering features..."):
        macro_df = macro_map.get(ticker) if macro_map else None

        # Get regime labels for this ticker
        regime_labels = None
        if regime_clf is not None:
            try:
                tmp = build_features(df, fit_regime=False)
                regime_labels = regime_clf.predict(tmp)
            except Exception:
                pass

        feat = build_features(df, macro_df=macro_df, regime_labels=regime_labels)
        if len(feat) >= 200:
            result[ticker] = feat

    total = sum(len(v) for v in result.values())
    console.print(f"[bold green]✓  Features ready for {len(result)} tickers[/bold green]")
    console.print(f"[bold green]  ✓  {total:,} total samples | {N_FEATURES} features per sample[/bold green]\n")
    return result
