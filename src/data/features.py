"""
src/data/features.py — cryp2me.ai Phase 5
Complete feature engineering: 35 features + correct 24/48/72h targets.

Key fixes vs Phase 4:
  - Targets use shift(-24), shift(-48), shift(-72) — NOT shift(-1/-2/-3)
  - Standardisation loop uses correct variable name
  - FEATURE_COLS and N_FEATURES exported at module level
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# ── Feature column registry ───────────────────────────────────────────────────

TECHNICAL_COLS = [
    "open_pct", "high_pct", "low_pct", "close_pct",
    "volume_norm", "volume_ratio", "vol_divergence",
    "buy_sell_ratio", "vol_momentum", "obv_norm",
    "ema10_dist", "ema20_dist", "ema34_dist",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "atr14_norm", "bb_width",
    "adx14", "plus_di", "minus_di",
]  # 22

MACRO_COLS = [
    "dxy_pct", "gold_pct", "spx_pct",
    "yield_10y_chg", "yield_curve",
    "vix_level", "vix_chg",
    "funding_rate", "oi_change",
]  # 9

REGIME_COLS = [
    "regime_0", "regime_1", "regime_2", "regime_3",
]  # 4

FEATURE_COLS    = TECHNICAL_COLS + MACRO_COLS + REGIME_COLS  # 35
TARGET_REG_COLS = ["ret1", "ret2", "ret3"]
TARGET_CLS_COLS = ["dir1", "dir2", "dir3"]
N_FEATURES      = len(FEATURE_COLS)  # 35


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _ema(series: np.ndarray, period: int) -> np.ndarray:
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
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    out    = np.full(len(tr), np.nan)
    out[period - 1] = tr[:period].mean()
    k = 1.0 / period
    for i in range(period, len(tr)):
        out[i] = tr[i] * k + out[i - 1] * (1 - k)
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n      = len(close)
    prev_h = np.roll(high,  1); prev_h[0] = high[0]
    prev_l = np.roll(low,   1); prev_l[0] = low[0]
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    up     = high  - prev_h
    down   = prev_l - low
    pdm    = np.where((up > down) & (up > 0), up, 0.0)
    mdm    = np.where((down > up) & (down > 0), down, 0.0)
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))

    atr14  = np.full(n, np.nan)
    pdm14  = np.full(n, np.nan)
    mdm14  = np.full(n, np.nan)
    atr14[period] = tr[1:period+1].sum()
    pdm14[period] = pdm[1:period+1].sum()
    mdm14[period] = mdm[1:period+1].sum()
    for i in range(period + 1, n):
        atr14[i] = atr14[i-1] - atr14[i-1]/period + tr[i]
        pdm14[i] = pdm14[i-1] - pdm14[i-1]/period + pdm[i]
        mdm14[i] = mdm14[i-1] - mdm14[i-1]/period + mdm[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr14 > 0, 100 * pdm14 / atr14, 0.0)
        mdi = np.where(atr14 > 0, 100 * mdm14 / atr14, 0.0)
        dx  = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    adx_s  = np.full(n, np.nan)
    start  = 2 * period
    if n > start:
        adx_s[start] = dx[period:start+1].mean()
        for i in range(start + 1, n):
            adx_s[i] = (adx_s[i-1] * (period - 1) + dx[i]) / period

    out_pdi = np.full(n, np.nan); out_mdi = np.full(n, np.nan)
    out_pdi[period:] = pdi[period:]; out_mdi[period:] = mdi[period:]
    return adx_s, out_pdi, out_mdi


# ── Main feature builder ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    macro_df:      Optional[pd.DataFrame] = None,
    regime_labels: Optional[np.ndarray]   = None,
    fit_regime:    bool = True,
) -> pd.DataFrame:
    """
    Build 35 features + targets from raw OHLCV data.

    Targets:
        ret1/dir1 = T+24h return/direction
        ret2/dir2 = T+48h return/direction
        ret3/dir3 = T+72h return/direction

    Args:
        df:            Raw OHLCV DataFrame [time, open, high, low, close, volume]
        macro_df:      Optional aligned macro features
        regime_labels: Optional pre-computed regime labels (0-3)
        fit_regime:    Fit regime classifier if labels not provided

    Returns:
        DataFrame with FEATURE_COLS + TARGET_*_COLS + time + close
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    t = df["time"].values

    # ── Price ─────────────────────────────────────────────────────────────────
    prev_c    = np.roll(c, 1); prev_c[0] = c[0]
    open_pct  = (o - prev_c) / (prev_c + 1e-10)
    high_pct  = (h - prev_c) / (prev_c + 1e-10)
    low_pct   = (l - prev_c) / (prev_c + 1e-10)
    close_pct = (c - prev_c) / (prev_c + 1e-10)

    # ── Volume ────────────────────────────────────────────────────────────────
    v_ma30 = _ema(v, 30)
    with np.errstate(divide='ignore', invalid='ignore'):
        volume_norm  = np.log1p(v)
        volume_ratio = np.where(v_ma30 > 0, v / v_ma30, 1.0)

    hl_range       = np.where((h - l) > 1e-10, h - l, 1e-10)
    buy_vol        = v * ((c - l) / hl_range)
    sell_vol       = np.where(v - buy_vol > 1e-10, v - buy_vol, 1e-10)
    buy_sell_ratio = np.log1p(np.clip(buy_vol / sell_vol, 0, 10))

    vol_dir      = np.sign(np.diff(v, prepend=v[0]))
    price_dir    = np.sign(close_pct)
    vol_divergence = (price_dir * vol_dir).astype(float)

    v_ma20       = _ema(volume_ratio, 20)
    vol_momentum = volume_ratio - np.where(np.isnan(v_ma20), 1.0, v_ma20)

    obv      = np.cumsum(np.sign(close_pct) * v)
    obv_std  = pd.Series(obv).rolling(168).std().values
    obv_mean = pd.Series(obv).rolling(168).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        obv_norm = np.where(obv_std > 1e-8, (obv - obv_mean) / obv_std, 0.0)

    # ── EMAs ──────────────────────────────────────────────────────────────────
    ema10 = _ema(c, 10); ema20 = _ema(c, 20); ema34 = _ema(c, 34)
    with np.errstate(divide='ignore', invalid='ignore'):
        ema10_dist = (c - ema10) / (c + 1e-10)
        ema20_dist = (c - ema20) / (c + 1e-10)
        ema34_dist = (c - ema34) / (c + 1e-10)

    # ── Momentum ──────────────────────────────────────────────────────────────
    rsi14 = _rsi(c, 14) / 100.0

    ema12     = _ema(c, 12); ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    k9        = 2.0 / 10
    macd_sig  = np.full(len(c), np.nan)
    valid     = np.where(~np.isnan(macd_line))[0]
    if len(valid) >= 9:
        si = valid[8]
        macd_sig[si] = np.nanmean(macd_line[valid[0]:si+1])
        for i in range(si + 1, len(c)):
            if not np.isnan(macd_line[i]):
                macd_sig[i] = macd_line[i] * k9 + macd_sig[i-1] * (1 - k9)
    macd_hist_arr = macd_line - macd_sig
    with np.errstate(divide='ignore', invalid='ignore'):
        macd_norm  = macd_line / (c + 1e-10)
        msig_norm  = macd_sig  / (c + 1e-10)
        mhist_norm = macd_hist_arr / (c + 1e-10)

    # ── Volatility ────────────────────────────────────────────────────────────
    atr14 = _atr(h, l, c, 14)
    with np.errstate(divide='ignore', invalid='ignore'):
        atr14_norm = atr14 / (c + 1e-10)

    bb_mid   = _ema(c, 20)
    roll_std = pd.Series(c).rolling(20).std().values
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = np.where(bb_mid > 0, 2 * roll_std / bb_mid, 0.0)

    # ── Trend ─────────────────────────────────────────────────────────────────
    adx14_arr, plus_di_arr, minus_di_arr = _adx(h, l, c, 14)
    adx14_norm    = adx14_arr    / 100.0
    plus_di_norm  = plus_di_arr  / 100.0
    minus_di_norm = minus_di_arr / 100.0

    # ── Targets — CORRECT: 24/48/72 hour horizons ─────────────────────────────
    # IMPORTANT: shift(-24) means "what is the price 24 hours from now"
    # This is intentionally looking forward — these are the labels we predict
    ret1 = (np.roll(c, -24) - c) / (c + 1e-10)
    ret2 = (np.roll(c, -48) - c) / (c + 1e-10)
    ret3 = (np.roll(c, -72) - c) / (c + 1e-10)
    ret1[-24:] = np.nan
    ret2[-48:] = np.nan
    ret3[-72:] = np.nan
    dir1 = (ret1 > 0).astype(float)
    dir2 = (ret2 > 0).astype(float)
    dir3 = (ret3 > 0).astype(float)

    # ── Assemble ──────────────────────────────────────────────────────────────
    feat = pd.DataFrame({
        "time": t, "close": c,
        "open_pct":       open_pct,   "high_pct":    high_pct,
        "low_pct":        low_pct,    "close_pct":   close_pct,
        "volume_norm":    volume_norm, "volume_ratio": volume_ratio,
        "vol_divergence": vol_divergence, "buy_sell_ratio": buy_sell_ratio,
        "vol_momentum":   vol_momentum,   "obv_norm":      obv_norm,
        "ema10_dist":     ema10_dist,  "ema20_dist":  ema20_dist,
        "ema34_dist":     ema34_dist,
        "rsi14":          rsi14,
        "macd":           macd_norm,   "macd_signal": msig_norm,
        "macd_hist":      mhist_norm,
        "atr14_norm":     atr14_norm,  "bb_width":    bb_width,
        "adx14":          adx14_norm,  "plus_di":     plus_di_norm,
        "minus_di":       minus_di_norm,
        "ret1": ret1, "ret2": ret2, "ret3": ret3,
        "dir1": dir1, "dir2": dir2, "dir3": dir3,
    })

    # Drop NaN warmup rows (keep rows where targets exist)
    warmup_cols = [c for c in feat.columns if c not in ["time", "close"]]
    feat = feat.dropna(subset=warmup_cols).reset_index(drop=True)

    # ── Macro features ────────────────────────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        for col in MACRO_COLS:
            if col in macro_df.columns:
                vals = macro_df[col].values
                feat[col] = vals[-len(feat):] if len(vals) >= len(feat) else 0.0
            else:
                feat[col] = 0.0
    else:
        for col in MACRO_COLS:
            feat[col] = 0.0

    # ── Regime features ───────────────────────────────────────────────────────
    if regime_labels is not None:
        labels = regime_labels
    elif fit_regime:
        try:
            from src.models.regime_classifier import RegimeClassifier
            rc = RegimeClassifier()
            rc.fit(feat)
            labels = rc.predict(feat)
        except Exception:
            labels = np.zeros(len(feat), dtype=int)
    else:
        labels = np.zeros(len(feat), dtype=int)

    if len(labels) > len(feat):
        labels = labels[-len(feat):]
    elif len(labels) < len(feat):
        labels = np.pad(labels, (len(feat) - len(labels), 0), constant_values=0)

    for r in range(4):
        feat[f"regime_{r}"] = (labels == r).astype(float)

    # ── Standardise features (z-score) ───────────────────────────────────────
    # NOTE: loop variable is `feat_col` to avoid shadowing the `feat` DataFrame
    for feat_col in FEATURE_COLS:
        if feat_col in feat.columns:
            mu  = feat[feat_col].mean()
            sig = feat[feat_col].std()
            if sig > 1e-8:
                feat[feat_col] = (feat[feat_col] - mu) / sig

    return feat


def build_all(
    raw_data:  Dict[str, pd.DataFrame],
    macro_map: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build features for all tickers with shared regime classifier."""
    from rich.console import Console
    from rich.progress import track
    console = Console()

    result     = {}
    regime_clf = None

    console.print("\n[bold cyan]⚙  Building features[/bold cyan]")

    # Fit regime classifier on combined data
    try:
        from src.models.regime_classifier import RegimeClassifier
        console.print("  🎯 Fitting regime classifier on all tickers...")
        tech_frames = []
        for df in raw_data.values():
            tmp = build_features(df, fit_regime=False)
            if len(tmp) > 200:
                tech_frames.append(tmp)
        if tech_frames:
            all_tech   = pd.concat(tech_frames, ignore_index=True)
            regime_clf = RegimeClassifier(
                cache_path=__import__("pathlib").Path("models/regime_classifier.pkl")
            )
            regime_clf.fit(all_tech)
            console.print("  ✓  Regime classifier fitted")
    except Exception as e:
        console.print(f"  ⚠  Regime classifier unavailable: {e}")

    for ticker, df in track(raw_data.items(), description="Engineering features..."):
        macro_df      = macro_map.get(ticker) if macro_map else None
        regime_labels = None

        if regime_clf is not None:
            try:
                tmp           = build_features(df, fit_regime=False)
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
