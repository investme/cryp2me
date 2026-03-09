"""
src/data/collector.py
─────────────────────
Downloads OHLCV from Binance for all configured tickers.
Caches raw data to disk to avoid re-downloading.
"""

import asyncio
import httpx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import track

console = Console()

BINANCE_BASE   = "https://api.binance.com/api/v3"
BINANCE_KLINES = f"{BINANCE_BASE}/klines"
MAX_CANDLES    = 1000   # Binance hard limit per request


async def _fetch_klines_page(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str,
    start_ms: int,
    limit: int = 1000,
) -> list:
    resp = await client.get(
        BINANCE_KLINES,
        params={"symbol": symbol, "interval": interval,
                "startTime": start_ms, "limit": limit},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_full_history(
    ticker: str,
    interval: str = "1d",
    lookback_days: int = 1500,
) -> Optional[pd.DataFrame]:
    """
    Fetch full OHLCV history using paginated requests.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    Returns None if ticker is not found on Binance.
    """
    now_ms    = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms  = now_ms - lookback_days * 24 * 3600 * 1000

    # Try quote currencies in order
    symbol = None
    for quote in ["USDT", "BUSD", "BTC"]:
        sym = f"{ticker.upper()}{quote}"
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(
                    BINANCE_KLINES,
                    params={"symbol": sym, "interval": interval, "limit": 1},
                    timeout=10.0,
                )
                if r.status_code == 200 and r.json():
                    symbol = sym
                    break
            except Exception:
                continue

    if symbol is None:
        console.print(f"  [yellow]⚠  {ticker}: not found on Binance — skipping[/yellow]")
        return None

    # Paginate through history
    all_rows = []
    cursor   = start_ms

    async with httpx.AsyncClient() as client:
        while cursor < now_ms:
            try:
                rows = await _fetch_klines_page(client, symbol, interval, cursor)
            except Exception as e:
                console.print(f"  [red]✗  {ticker}: fetch error — {e}[/red]")
                break
            if not rows:
                break
            all_rows.extend(rows)
            cursor = int(rows[-1][6]) + 1   # closeTime + 1ms
            await asyncio.sleep(0.05)        # respect rate limits

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.columns = ["time", "open", "high", "low", "close", "volume"]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = (df["time"] // 1000).astype(int)   # → Unix seconds

    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df


def collect_all(
    tickers: list,
    interval: str = "1d",
    lookback_days: int = 1500,
    cache_dir: Path = Path("data/cache"),
) -> dict[str, pd.DataFrame]:
    """
    Sync wrapper. Fetches all tickers, skips already-cached ones.
    Returns dict: ticker → DataFrame.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    console.print(f"\n[bold cyan]📡 Collecting OHLCV data — {len(tickers)} tickers[/bold cyan]")

    for ticker in track(tickers, description="Downloading..."):
        cache_file = cache_dir / f"{ticker}_{interval}.parquet"

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            # Refresh if last candle is >1 day old
            last_ts = df["time"].iloc[-1]
            age_days = (pd.Timestamp.now(tz="UTC").timestamp() - last_ts) / 86400
            if age_days < 2:
                results[ticker] = df
                continue

        df = asyncio.run(fetch_full_history(ticker, interval, lookback_days))
        if df is not None and len(df) >= 100:
            df.to_parquet(cache_file, index=False)
            results[ticker] = df
            console.print(f"  [green]✓[/green]  {ticker}: {len(df)} candles")
        else:
            console.print(f"  [yellow]⚠[/yellow]  {ticker}: insufficient data")

    console.print(f"\n[bold green]✓ Collected {len(results)}/{len(tickers)} tickers[/bold green]\n")
    return results
