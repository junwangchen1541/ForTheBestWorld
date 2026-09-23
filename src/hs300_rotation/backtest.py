from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .strategy import choose_holdings, prepare_features, score_prepared


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def _commission(value: float, rate: float, minimum: float) -> float:
    return max(value * rate, minimum) if value > 0 else 0.0


def _metrics(equity: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if equity.empty:
        return {}
    nav = equity["equity"]
    returns = nav.pct_change().dropna()
    years = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days / 365.25, 1 / 365.25)
    total_return = nav.iloc[-1] / initial_cash - 1
    annual_return = (nav.iloc[-1] / initial_cash) ** (1 / years) - 1
    drawdown = nav / nav.cummax() - 1
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    return {
        "final_equity": float(nav.iloc[-1]),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }


def run_backtest(
    data: pd.DataFrame,
    config: AppConfig,
    prepared_features: pd.DataFrame | None = None,
    score_provider: Callable[[pd.Timestamp], pd.DataFrame] | None = None,
) -> BacktestResult:
    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date)
    data = data[(data["date"] >= start) & (data["date"] <= end)].copy()
    dates = pd.Index(data["date"].drop_duplicates().sort_values())
    minimum_dates = 2 if score_provider is not None else config.strategy.minimum_history + 2
    if len(dates) < minimum_dates:
        raise ValueError("Not enough trading days for configured minimum_history")
    if prepared_features is None:
        prepared_features = prepare_features(data)
    else:
        prepared_features = prepared_features[
            (prepared_features["date"] >= start) & (prepared_features["date"] <= end)
        ]

    week_key = data["date"].dt.to_period("W-FRI")
    signal_dates = set(data.groupby(week_key)["date"].max())
    next_date = {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}
    scheduled: dict[pd.Timestamp, tuple[list[str], list[str], pd.Timestamp, list[str]]] = {}
    positions: dict[str, int] = {}
    last_close: dict[str, float] = {}
    cash = config.portfolio.initial_cash
    equity_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for date in dates:
        day = data[data["date"] == date].set_index("symbol")
        last_close.update(day["close"].astype(float).to_dict())
        if date in scheduled:
            sells, buys, signal_date, alternatives = scheduled.pop(date)
            for symbol in sells:
                shares = positions.get(symbol, 0)
                if shares <= 0 or symbol not in day.index:
                    continue
                price = float(day.at[symbol, "open"])
                value = shares * price
                fee = _commission(value, config.portfolio.commission_rate, config.portfolio.minimum_commission)
                tax = value * config.portfolio.stamp_duty_rate
                cash += value - fee - tax
                del positions[symbol]
                trade_rows.append({"signal_date": signal_date, "date": date, "symbol": symbol, "side": "SELL", "shares": shares, "price": price, "fee": fee + tax})

            open_value = sum(shares * last_close[symbol] for symbol, shares in positions.items())
            target_value = (cash + open_value) / config.portfolio.holding_count
            buy_target = max(0, config.portfolio.holding_count - len(positions))
            buy_order = list(dict.fromkeys([*buys, *alternatives]))
            bought = 0
            for symbol in buy_order:
                if bought >= buy_target:
                    break
                if symbol in positions:
                    continue
                if symbol not in day.index:
                    continue
                price = float(day.at[symbol, "open"])
                affordable = min(target_value, cash - config.portfolio.minimum_commission)
                shares = int(affordable / price / config.portfolio.lot_size) * config.portfolio.lot_size
                value = shares * price
                fee = _commission(value, config.portfolio.commission_rate, config.portfolio.minimum_commission)
                while shares > 0 and value + fee > cash:
                    shares -= config.portfolio.lot_size
                    value = shares * price
                    fee = _commission(value, config.portfolio.commission_rate, config.portfolio.minimum_commission)
                if shares <= 0:
                    continue
                cash -= value + fee
                positions[symbol] = positions.get(symbol, 0) + shares
                bought += 1
                trade_rows.append({"signal_date": signal_date, "date": date, "symbol": symbol, "side": "BUY", "shares": shares, "price": price, "fee": fee})

        close_value = sum(shares * last_close[symbol] for symbol, shares in positions.items())
        equity_rows.append({"date": date, "cash": cash, "market_value": close_value, "equity": cash + close_value, "holding_count": len(positions)})

        if date in signal_dates and date in next_date:
            scores = (
                score_provider(date)
                if score_provider is not None
                else score_prepared(prepared_features, date, config.strategy)
            )
            _, sells, buys = choose_holdings(scores, set(positions), config.portfolio)
            bought_candidates = set(buys)
            held_candidates = set(positions) - set(sells)
            alternatives = [
                str(symbol)
                for symbol in scores["symbol"]
                if str(symbol) not in bought_candidates and str(symbol) not in held_candidates
            ]
            scheduled[next_date[date]] = (sells, buys, date, alternatives)

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows, columns=["signal_date", "date", "symbol", "side", "shares", "price", "fee"])
    return BacktestResult(equity=equity, trades=trades, metrics=_metrics(equity, config.portfolio.initial_cash))
