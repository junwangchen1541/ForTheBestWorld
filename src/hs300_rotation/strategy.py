from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig, StrategyConfig

FEATURE_COLUMNS = ["momentum_20", "momentum_60", "momentum_120", "volatility_20", "drawdown_60"]


def prepare_features(history: pd.DataFrame) -> pd.DataFrame:
    """Calculate reusable price features without applying strategy weights."""
    ordered = history.sort_values(["symbol", "date"]).copy()
    grouped = ordered.groupby("symbol", sort=False)
    ordered["momentum_20"] = grouped["close"].pct_change(20, fill_method=None)
    ordered["momentum_60"] = grouped["close"].pct_change(60, fill_method=None)
    ordered["momentum_120"] = grouped["close"].pct_change(120, fill_method=None)
    daily_return = grouped["close"].pct_change(fill_method=None)
    ordered["volatility_20"] = daily_return.groupby(ordered["symbol"]).transform(
        lambda s: s.rolling(20).std() * np.sqrt(252)
    )
    rolling_high = grouped["close"].transform(lambda s: s.rolling(60).max())
    ordered["drawdown_60"] = ordered["close"] / rolling_high - 1.0
    ordered["history_count"] = grouped.cumcount() + 1
    return ordered


def score_prepared(
    prepared: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Score one cross-section from a precomputed feature panel."""
    latest = prepared[prepared["date"] == signal_date].copy()
    if latest.empty:
        return latest.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=int))

    latest = latest[
        (latest["history_count"] >= config.minimum_history)
        & (latest["close"] >= config.minimum_price)
        & (latest["amount"] >= config.minimum_daily_amount)
    ].dropna(subset=FEATURE_COLUMNS)
    if latest.empty:
        return latest.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=int))

    weights = {
        "momentum_20": config.momentum_20_weight,
        "momentum_60": config.momentum_60_weight,
        "momentum_120": config.momentum_120_weight,
        "volatility_20": config.low_volatility_weight,
        "drawdown_60": config.low_drawdown_weight,
    }
    latest["score"] = 0.0
    for column, weight in weights.items():
        ascending = column != "volatility_20"
        percentile = latest[column].rank(pct=True, ascending=ascending)
        latest["score"] += weight * percentile
    latest = latest.sort_values(["score", "symbol"], ascending=[False, True])
    latest["rank"] = np.arange(1, len(latest) + 1)
    return latest.reset_index(drop=True)


def score_universe(history: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Score symbols using information available at the final date in history."""
    prepared = prepare_features(history)
    return score_prepared(prepared, prepared["date"].max(), config)


def choose_holdings(
    scores: pd.DataFrame,
    current: set[str],
    config: PortfolioConfig,
) -> tuple[list[str], list[str], list[str]]:
    ranked = scores["symbol"].astype(str).tolist()
    if not current:
        selected = ranked[: config.holding_count]
        return selected, [], selected

    eligible_current = [symbol for symbol in ranked if symbol in current]
    forced_sells = sorted(current - set(eligible_current))
    held_worst_first = list(reversed(eligible_current))
    outsiders = [symbol for symbol in ranked if symbol not in current]

    # Forced exits can exceed the normal turnover cap because an ineligible name
    # should not remain in the target portfolio. Additional replacements require
    # the incoming candidate to outrank the corresponding current holding.
    replacement_count = max(config.min_replacements, len(forced_sells))
    replacement_count = min(replacement_count, len(current), len(outsiders))
    while replacement_count < min(config.max_replacements, len(current), len(outsiders)):
        held_index = replacement_count - len(forced_sells)
        if held_index >= len(held_worst_first):
            break
        incoming = outsiders[replacement_count]
        outgoing = held_worst_first[held_index]
        incoming_score = float(scores.loc[scores["symbol"] == incoming, "score"].iloc[0])
        outgoing_score = float(scores.loc[scores["symbol"] == outgoing, "score"].iloc[0])
        if incoming_score <= outgoing_score:
            break
        replacement_count += 1

    sells = forced_sells.copy()
    for symbol in held_worst_first:
        if len(sells) >= replacement_count:
            break
        if symbol not in sells:
            sells.append(symbol)
    buys = outsiders[: len(sells)]
    selected_set = (current - set(sells)) | set(buys)

    if len(selected_set) < config.holding_count:
        for symbol in ranked:
            if symbol not in selected_set:
                selected_set.add(symbol)
                buys.append(symbol)
            if len(selected_set) == config.holding_count:
                break
    selected = [symbol for symbol in ranked if symbol in selected_set]
    return selected, sells, buys
