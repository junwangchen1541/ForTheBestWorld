from dataclasses import replace

import pandas as pd
from test_strategy import market_data

from hs300_rotation.backtest import run_backtest
from hs300_rotation.config import AppConfig, BacktestConfig, PortfolioConfig, StrategyConfig


def test_backtest_uses_next_session_and_respects_lots() -> None:
    data = market_data(symbols=12, days=140)
    config = AppConfig(
        portfolio=PortfolioConfig(initial_cash=200_000, holding_count=3, min_replacements=1, max_replacements=1),
        strategy=replace(StrategyConfig(), minimum_history=121),
        backtest=BacktestConfig(start_date="2024-01-01", end_date="2025-01-01"),
    )
    result = run_backtest(data, config)
    assert not result.trades.empty
    assert (result.trades["date"] > result.trades["signal_date"]).all()
    assert (result.trades["shares"] % 100 == 0).all()
    assert result.metrics["final_equity"] > 0


def test_backtest_fills_from_next_ranked_names_when_top_pick_is_unaffordable() -> None:
    data = market_data(symbols=12, days=10)
    expensive_symbol = "000012"
    data.loc[data["symbol"] == expensive_symbol, ["open", "close"]] = 300.0
    config = AppConfig(
        portfolio=PortfolioConfig(initial_cash=200_000, holding_count=10),
        backtest=BacktestConfig(start_date="2024-01-01", end_date="2025-01-01"),
    )

    def scores(date):
        symbols = sorted(data["symbol"].unique(), reverse=True)
        return pd.DataFrame({"symbol": symbols, "score": range(len(symbols), 0, -1)})

    result = run_backtest(data, config, score_provider=scores)
    last_trade_date = result.trades["date"].max()
    held = result.trades[
        (result.trades["date"] <= last_trade_date)
        & (result.trades["side"] == "BUY")
    ]["symbol"].nunique()
    assert held == 10
    assert expensive_symbol not in set(result.trades.loc[result.trades["side"] == "BUY", "symbol"])
