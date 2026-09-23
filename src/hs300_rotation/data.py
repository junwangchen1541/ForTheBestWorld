from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

import pandas as pd

REQUIRED_COLUMNS = {"date", "symbol", "open", "close", "volume", "amount"}


def _fetch_tx_history(
    symbol: str,
    name: str,
    start_date: str,
    end_date: str,
    client: object,
) -> pd.DataFrame:
    market_prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    tx_symbol = f"{market_prefix}{symbol}"
    frames: list[pd.DataFrame] = []
    for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
        response = client.get(
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
            params={
                "_var": f"kline_dayqfq{year}",
                "param": f"{tx_symbol},day,{year}-01-01,{year}-12-31,640,qfq",
                "r": "0.8205512681390605",
            },
            timeout=30,
        )
        response.raise_for_status()
        decoded = json.loads(response.text[response.text.find("={") + 1 :])
        data = decoded.get("data", {}).get(tx_symbol, {})
        rows = data.get("qfqday") or data.get("day") or []
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True)
    history = history.iloc[:, [0, 1, 2, 5, 7]]
    history.columns = ["date", "open", "close", "volume", "amount"]
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history[["open", "close", "volume", "amount"]] = history[
        ["open", "close", "volume", "amount"]
    ].apply(pd.to_numeric, errors="coerce")
    if not tx_symbol.startswith(("sh688", "sz399", "sh000", "sz000")):
        history["volume"] *= 100
    history["amount"] *= 10_000
    history = history[
        (history["date"] >= pd.Timestamp(start_date))
        & (history["date"] <= pd.Timestamp(end_date))
    ]
    history = history.drop_duplicates("date").sort_values("date")
    if history.empty:
        return pd.DataFrame()
    history["symbol"] = symbol
    history["name"] = name
    return history[["date", "symbol", "name", "open", "close", "volume", "amount"]]


def load_market_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["symbol"] = frame["symbol"].str.zfill(6)
    numeric = ["open", "close", "volume", "amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "open", "close"])
    frame = frame.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"])
    if (frame[["open", "close"]] <= 0).any().any():
        raise ValueError("Open and close prices must be positive")
    return frame.reset_index(drop=True)


def download_current_universe(
    start: str,
    end: str,
    output: str | Path,
    universe: str = "hs300",
) -> Path:
    """Download adjusted Tencent daily history for today's A-share or CSI 300 universe.

    Both modes use current constituents, so they are convenient for research but
    not point-in-time-clean historical universes.
    """
    try:
        import akshare as ak
        import requests
    except ImportError as exc:
        raise RuntimeError("Install the data extra first: pip install -e .[data]") from exc

    start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end).strftime("%Y-%m-%d")

    if universe == "hs300":
        members = ak.index_stock_cons(symbol="000300")
        code_column = next((c for c in ("品种代码", "成分券代码", "symbol") if c in members), None)
        name_column = next((c for c in ("品种名称", "成分券名称", "name") if c in members), None)
    elif universe == "all-a":
        members = ak.stock_zh_a_spot_tx()
        code_column = next((c for c in ("code", "symbol") if c in members), None)
        name_column = next((c for c in ("name",) if c in members), None)
    else:
        raise ValueError("universe must be 'hs300' or 'all-a'")
    if code_column is None:
        raise RuntimeError(f"Unexpected AkShare constituent columns: {list(members.columns)}")

    names = {}
    if name_column:
        names = dict(
            zip(
                members[code_column].astype(str).str.replace(r"^(sh|sz)", "", regex=True).str.zfill(6),
                members[name_column].astype(str),
                strict=True,
            )
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_symbols: set[str] = set()
    if destination.exists():
        existing_symbols = set(pd.read_csv(destination, usecols=["symbol"], dtype=str)["symbol"])

    sessions = local()

    def fetch_symbol(symbol: str) -> pd.DataFrame:
        if not hasattr(sessions, "client"):
            sessions.client = requests.Session()
        for attempt in range(3):
            try:
                return _fetch_tx_history(
                    symbol,
                    names.get(symbol, symbol),
                    start_date,
                    end_date,
                    sessions.client,
                )
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return pd.DataFrame()

    symbols = [
        raw_symbol.removeprefix("sh").removeprefix("sz").zfill(6)
        for raw_symbol in members[code_column].astype(str)
        if raw_symbol.removeprefix("sh").removeprefix("sz").zfill(6) not in existing_symbols
    ]
    total = len(symbols)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_symbol, symbol): symbol for symbol in symbols}
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = future.result()
            except (requests.RequestException, KeyError, ValueError, TypeError, IndexError) as exc:
                print(f"Skipped {symbol}: {exc}")
                continue
            if not frame.empty:
                frame.sort_values(["date", "symbol"]).to_csv(
                    destination,
                    mode="a",
                    header=not destination.exists(),
                    index=False,
                )
                existing_symbols.add(symbol)
            if completed % 25 == 0 or completed == total:
                print(f"Downloaded {completed}/{total} remaining symbols into {destination}")
    if not destination.exists():
        raise RuntimeError("Tencent returned no price history")
    return destination


def download_current_hs300(start: str, end: str, output: str | Path) -> Path:
    """Backward-compatible CSI 300 download helper."""
    return download_current_universe(start, end, output, universe="hs300")
