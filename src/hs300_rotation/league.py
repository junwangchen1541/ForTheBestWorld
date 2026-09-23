from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import AppConfig
from .models import FittedExperiment, fit_experiment, make_labeled_features


@dataclass(frozen=True)
class ExperimentProfile:
    model_id: str
    model_name: str
    thesis: str


MODEL_PROFILES = [
    ExperimentProfile("tree", "梯度提升树", "在数值技术特征上拟合未来收益"),
    ExperimentProfile("laya", "Laya + 校准", "Laya 决策评分，经训练股票池收益校准"),
    ExperimentProfile("mlp", "前馈神经网络", "多层感知机拟合未来收益"),
]
POOL_PROFILES = [("all_a", "全 A 股训练"), ("hs300", "沪深300训练")]


def _positions_from_trades(trades: pd.DataFrame) -> dict[str, int]:
    positions: dict[str, int] = {}
    for trade in trades.itertuples(index=False):
        direction = 1 if trade.side == "BUY" else -1
        positions[trade.symbol] = positions.get(trade.symbol, 0) + direction * int(trade.shares)
    return {symbol: shares for symbol, shares in positions.items() if shares > 0}


def _experiment_payload(
    experiment: FittedExperiment,
    result: BacktestResult,
    evaluation_data: pd.DataFrame,
    thesis: str,
) -> dict[str, object]:
    final_date = result.equity["date"].iloc[-1]
    latest = (
        evaluation_data[evaluation_data["date"] <= final_date]
        .sort_values("date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .set_index("symbol")
    )
    scores = experiment.score_on(final_date).set_index("symbol")
    positions = _positions_from_trades(result.trades)
    total_value = sum(
        shares * float(latest.at[symbol, "close"])
        for symbol, shares in positions.items()
        if symbol in latest.index
    )
    holdings = []
    for symbol, shares in positions.items():
        if symbol not in latest.index:
            continue
        close = float(latest.at[symbol, "close"])
        name = latest.at[symbol, "name"] if "name" in latest.columns else symbol
        holdings.append(
            {
                "symbol": symbol,
                "name": str(name),
                "shares": shares,
                "close": round(close, 3),
                "market_value": round(shares * close, 2),
                "weight": round(shares * close / total_value, 6) if total_value else 0.0,
                "score": round(float(scores.at[symbol, "score"]), 6)
                if symbol in scores.index
                else None,
                "rank": int(scores.at[symbol, "rank"]) if symbol in scores.index else None,
            }
        )
    holdings.sort(key=lambda item: item["market_value"], reverse=True)

    equity = result.equity.copy()
    equity["nav"] = equity["equity"] / equity["equity"].iloc[0]
    stride = max(1, len(equity) // 260)
    equity_points = [
        {"date": row.date.strftime("%Y-%m-%d"), "value": round(float(row.nav), 6)}
        for row in equity.iloc[::stride].itertuples(index=False)
    ]
    if equity_points[-1]["date"] != final_date.strftime("%Y-%m-%d"):
        equity_points.append(
            {"date": final_date.strftime("%Y-%m-%d"), "value": round(float(equity["nav"].iloc[-1]), 6)}
        )

    trades = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "side": row.side,
            "symbol": row.symbol,
            "name": str(latest.at[row.symbol, "name"])
            if row.symbol in latest.index and "name" in latest.columns
            else row.symbol,
            "shares": int(row.shares),
            "price": round(float(row.price), 3),
            "fee": round(float(row.fee), 2),
        }
        for row in result.trades.itertuples(index=False)
    ]
    return {
        "id": f"{experiment.model_id}_{experiment.pool_id}",
        "model_id": experiment.model_id,
        "pool_id": experiment.pool_id,
        "name": f"{experiment.model_name} · {experiment.pool_name}",
        "model_name": experiment.model_name,
        "pool_name": experiment.pool_name,
        "thesis": thesis,
        "metrics": {**result.metrics, "trade_count": len(result.trades)},
        "training": {
            "samples": experiment.training_rows,
            "start_date": experiment.training_start,
            "end_date": experiment.training_end,
            "target_horizon_days": experiment.target_horizon_days,
            "validation_rmse": experiment.validation_rmse
            if np.isfinite(experiment.validation_rmse)
            else None,
            "laya_version": experiment.laya_version,
            "weights_trainable": experiment.model_id != "laya",
        },
        "equity": equity_points,
        "holdings": holdings,
        "trades": list(reversed(trades)),
    }


def run_league(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    config: AppConfig,
    hs300_symbols: set[str] | None = None,
) -> dict[str, object]:
    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date)
    cutoff = pd.Timestamp(config.model.training_end_date)
    test_start = max(start, cutoff + pd.offsets.Day(1))
    evaluation_history = evaluation_data[evaluation_data["date"] <= end].copy()
    evaluation_window = evaluation_history[evaluation_history["date"] >= test_start].copy()
    if evaluation_window.empty:
        raise ValueError("No evaluation data falls inside the configured backtest window")
    universe = set(hs300_symbols or evaluation_history["symbol"].astype(str).unique())
    evaluation_history = evaluation_history[evaluation_history["symbol"].isin(universe)]
    evaluation_window = evaluation_window[evaluation_window["symbol"].isin(universe)]
    if evaluation_window.empty:
        raise ValueError("Evaluation set is empty after applying the shared HS300 universe")

    all_symbols = set(training_data.loc[training_data["date"] <= cutoff, "symbol"].astype(str).unique())
    prepared_training = make_labeled_features(training_data, config.model.target_horizon_days)
    prepared_evaluation = make_labeled_features(evaluation_history, config.model.target_horizon_days)
    experiments = []
    profiles_by_id = {profile.model_id: profile for profile in MODEL_PROFILES}
    for model_id in ("tree", "laya", "mlp"):
        for pool_id, _pool_name in POOL_PROFILES:
            experiment = fit_experiment(
                training_data=training_data,
                evaluation_data=evaluation_history,
                pool=pool_id,
                model_id=model_id,
                hs300_symbols=universe,
                model_config=config.model,
                prepared_training=prepared_training,
                prepared_evaluation=prepared_evaluation,
            )
            profile_config = replace(
                config,
                backtest=replace(config.backtest, start_date=test_start.strftime("%Y-%m-%d")),
            )
            def model_scores(date: pd.Timestamp, fitted: FittedExperiment = experiment) -> pd.DataFrame:
                scores = fitted.score_on(date)
                if scores.empty:
                    return scores
                market = evaluation_history[evaluation_history["date"] == date].set_index("symbol")
                eligible = market.index[
                    (market["close"] >= config.strategy.minimum_price)
                    & (market["amount"] >= config.strategy.minimum_daily_amount)
                ]
                return scores[scores["symbol"].isin(eligible)].reset_index(drop=True)

            result = run_backtest(
                evaluation_window,
                profile_config,
                score_provider=model_scores,
            )
            experiments.append(
                _experiment_payload(
                    experiment,
                    result,
                    evaluation_history,
                    profiles_by_id[model_id].thesis,
                )
            )

    final_date = evaluation_window["date"].max()
    return {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(timespec="seconds"),
        "as_of_date": final_date.strftime("%Y-%m-%d"),
        "start_date": evaluation_window["date"].min().strftime("%Y-%m-%d"),
        "training_end_date": cutoff.strftime("%Y-%m-%d"),
        "evaluation_universe": "沪深300",
        "training_universe_sizes": {"all_a": len(all_symbols), "hs300": len(universe)},
        "model_matrix": [{"model_id": item.model_id, "model_name": item.model_name} for item in MODEL_PROFILES],
        "initial_cash": config.portfolio.initial_cash,
        "is_demo": False,
        "experiments": experiments,
        "strategies": experiments,
    }


def write_league_outputs(payload: dict[str, object], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    destination.write_text(f"window.COMPETITION_DATA={serialized};\n", encoding="utf-8")
    return destination
