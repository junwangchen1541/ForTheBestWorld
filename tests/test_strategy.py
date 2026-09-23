import numpy as np
import pandas as pd

from hs300_rotation.config import PortfolioConfig, StrategyConfig
from hs300_rotation.strategy import choose_holdings, score_universe


def market_data(symbols: int = 12, days: int = 140) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for number in range(symbols):
        symbol = f"{number + 1:06d}"
        growth = 0.0005 + number * 0.0002
        close = 10 * np.exp(np.arange(days) * growth)
        for date, price in zip(dates, close, strict=True):
            rows.append({"date": date, "symbol": symbol, "open": price, "close": price, "volume": 1_000_000, "amount": 50_000_000})
    return pd.DataFrame(rows)


def test_score_universe_ranks_stronger_trend_first() -> None:
    scores = score_universe(market_data(), StrategyConfig())
    assert scores.iloc[0]["symbol"] == "000012"
    assert scores.iloc[-1]["symbol"] == "000001"
    assert scores["rank"].tolist() == list(range(1, 13))


def test_choose_holdings_replaces_worst_current_name() -> None:
    scores = score_universe(market_data(), StrategyConfig())
    config = PortfolioConfig(holding_count=3, min_replacements=1, max_replacements=1)
    selected, sells, buys = choose_holdings(scores, {"000001", "000011", "000012"}, config)
    assert sells == ["000001"]
    assert buys == ["000010"]
    assert selected == ["000012", "000011", "000010"]


def test_choose_holdings_can_replace_two_outdated_names() -> None:
    scores = score_universe(market_data(), StrategyConfig())
    config = PortfolioConfig(holding_count=3, min_replacements=1, max_replacements=2)
    selected, sells, buys = choose_holdings(scores, {"000001", "000002", "000012"}, config)
    assert sells == ["000001", "000002"]
    assert buys == ["000011", "000010"]
    assert selected == ["000012", "000011", "000010"]
