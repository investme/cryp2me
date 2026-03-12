"""
src/data/collector.py — cryp2me.ai Phase 5
Load OHLCV data from local cache (parquet) or Binance API.

On Kaggle/Colab: always use --skip-download and pre-upload parquet cache.
On Windows local: will try Binance, falls back to yfinance if geo-blocked.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from rich.console import Console
from rich.progress import track

console = Console()

BINANCE_BASE = "https://api.binance.com/api/v3"


def _binance_klines(
    symbol:    str,
    interval:  str = "1h",
    start_ts:  int = None,
    end_ts:    int = None,
    limit:     int = 1000,
) -> Optional[pd.DataFrame]:
    """Fetch klines from Binance REST API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ts:
        params["startTime"] = int(start_ts)
    if end_ts:
        params["endTime"] = int(end_ts)

    try:
        r = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "n_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["time"]   = pd.to_datetime(df["time"].astype(float), unit="ms")
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        return None


def _fetch_binance_full(
    ticker:       str,
    interval:     str = "1h",
    lookback_days:int = 365 * 4,
) -> Optional[pd.DataFrame]:
    """Fetch full history from Binance in 1000-candle pages."""
    symbol  = f"{ticker}USDT"
    end_ts  = int(datetime.utcnow().timestamp() * 1000)
    start_dt= datetime.utcnow() - timedelta(days=lookback_days)
    start_ts= int(start_dt.timestamp() * 1000)

    all_rows = []
    cur_start = start_ts

    while cur_start < end_ts:
        df = _binance_klines(symbol, interval, start_ts=cur_start, limit=1000)
        if df is None or len(df) == 0:
            break
        all_rows.append(df)
        # Advance: last candle time + 1ms
        last_ts_ms = int(df["time"].iloc[-1].timestamp() * 1000) + 1
        if last_ts_ms <= cur_start:
            break
        cur_start = last_ts_ms
        if len(df) < 1000:
            break
        time.sleep(0.1)  # gentle rate limiting

    if not all_rows:
        return None

    full = pd.concat(all_rows, ignore_index=True)
    full = full.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return full


def _fetch_yfinance(
    ticker:       str,
    lookback_days:int = 365 * 4,
) -> Optional[pd.DataFrame]:
    """Fallback: fetch from yfinance (daily → resampled, less ideal)."""
    try:
        import yfinance as yf
        sym  = f"{ticker}-USD"
        start= (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df   = yf.download(sym, start=start, interval="1h", progress=False, auto_adjust=True)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"datetime": "time", "index": "time"})
        if "time" not in df.columns:
            df["time"] = df.index
        df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        return df.reset_index(drop=True)
    except Exception as e:
        console.print(f"    [red]yfinance failed for {ticker}: {e}[/red]")
        return None


def _load_from_cache(ticker: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    """Load from parquet cache file."""
    paths = [
        cache_dir / f"{ticker}USDT_1h.parquet",
        cache_dir / f"{ticker}_1h.parquet",
        cache_dir / f"{ticker.lower()}_1h.parquet",
        cache_dir / f"{ticker}.parquet",
    ]
    for p in paths:
        if p.exists():
            try:
                df = pd.read_parquet(p)
                if "time" not in df.columns and df.index.name in ["time", "datetime", "timestamp"]:
                    df = df.reset_index().rename(columns={df.index.name: "time"})
                df["time"] = pd.to_datetime(df["time"])
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                return df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
            except Exception as e:
                console.print(f"    [yellow]Cache read error {p.name}: {e}[/yellow]")
    return None


def _save_to_cache(df: pd.DataFrame, ticker: str, cache_dir: Path):
    """Save to parquet cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{ticker}USDT_1h.parquet"
    df.to_parquet(path, index=False)


def collect_ticker(
    ticker:        str,
    interval:      str  = "1h",
    lookback_days: int  = 365 * 4,
    cache_dir:     Path = Path("data/cache"),
    skip_download: bool = False,
    min_rows:      int  = 500,
) -> Optional[pd.DataFrame]:
    """
    Load one ticker. Priority:
      1. Cache (parquet) if exists
      2. Binance REST API
      3. yfinance fallback
    """
    # Try cache first
    cached = _load_from_cache(ticker, cache_dir)
    if cached is not None and len(cached) >= min_rows:
        return cached

    if skip_download:
        if cached is not None:
            console.print(f"    [yellow]⚠  {ticker}: only {len(cached)} rows in cache[/yellow]")
            return cached if len(cached) >= 100 else None
        return None

    # Try Binance
    console.print(f"    Fetching {ticker} from Binance...")
    df = _fetch_binance_full(ticker, interval, lookback_days)

    if df is not None and len(df) >= min_rows:
        _save_to_cache(df, ticker, cache_dir)
        return df

    # Fallback: yfinance
    console.print(f"    [yellow]{ticker}: Binance unavailable, trying yfinance...[/yellow]")
    df = _fetch_yfinance(ticker, lookback_days)
    if df is not None and len(df) >= min_rows:
        _save_to_cache(df, ticker, cache_dir)
        return df

    console.print(f"    [red]✗  {ticker}: no data available[/red]")
    return None


def collect_all(
    tickers:       List[str],
    interval:      str  = "1h",
    lookback_days: int  = 365 * 4,
    cache_dir:     Path = Path("data/cache"),
    skip_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Collect OHLCV data for all tickers.

    Returns dict: {ticker: df}
    Only includes tickers with sufficient data.
    """
    cache_dir = Path(cache_dir)
    console.print(f"\n[bold cyan]📥 Data Collection[/bold cyan]")
    console.print(f"  Tickers: {len(tickers)}  |  Interval: {interval}  |  "
                  f"Lookback: {lookback_days}d  |  "
                  f"{'Cache only' if skip_download else 'Download if missing'}\n")

    result     = {}
    failed     = []

    for ticker in track(tickers, description="Loading data..."):
        df = collect_ticker(
            ticker        = ticker,
            interval      = interval,
            lookback_days = lookback_days,
            cache_dir     = cache_dir,
            skip_download = skip_download,
        )
        if df is not None and len(df) >= 500:
            result[ticker] = df
        else:
            failed.append(ticker)

    console.print(f"\n[bold green]✓  Loaded {len(result)}/{len(tickers)} tickers[/bold green]")
    if failed:
        console.print(f"[yellow]  Skipped: {', '.join(failed)}[/yellow]")

    total = sum(len(v) for v in result.values())
    console.print(f"[bold green]  Total rows: {total:,}[/bold green]\n")
    return result
