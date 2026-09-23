from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class PortfolioConfig:
    initial_cash: float = 200_000.0
    holding_count: int = 10
    min_replacements: int = 1
    max_replacements: int = 2
    lot_size: int = 100
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    stamp_duty_rate: float = 0.0005


@dataclass(frozen=True)
class StrategyConfig:
    minimum_history: int = 121
    minimum_price: float = 2.0
    minimum_daily_amount: float = 20_000_000.0
    momentum_20_weight: float = 0.25
    momentum_60_weight: float = 0.30
    momentum_120_weight: float = 0.25
    low_volatility_weight: float = 0.10
    low_drawdown_weight: float = 0.10


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2099-12-31"
    benchmark_symbol: str = "000300"


@dataclass(frozen=True)
class ModelConfig:
    training_end_date: str = "2026-09-17"
    target_horizon_days: int = 20
    minimum_training_samples: int = 500
    laya_calibration_samples: int = 256


@dataclass(frozen=True)
class AppConfig:
    portfolio: PortfolioConfig = PortfolioConfig()
    strategy: StrategyConfig = StrategyConfig()
    backtest: BacktestConfig = BacktestConfig()
    model: ModelConfig = ModelConfig()


T = TypeVar("T")


def _section(cls: type[T], raw: dict[str, Any], name: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in [{name}]: {', '.join(sorted(unknown))}")
    return cls(**raw)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    config = AppConfig(
        portfolio=_section(PortfolioConfig, raw.get("portfolio", {}), "portfolio"),
        strategy=_section(StrategyConfig, raw.get("strategy", {}), "strategy"),
        backtest=_section(BacktestConfig, raw.get("backtest", {}), "backtest"),
        model=_section(ModelConfig, raw.get("model", {}), "model"),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    p = config.portfolio
    if p.initial_cash <= 0 or p.holding_count <= 0 or p.lot_size <= 0:
        raise ValueError("Cash, holding_count, and lot_size must be positive")
    if not 0 <= p.min_replacements <= p.max_replacements <= p.holding_count:
        raise ValueError("Require 0 <= min_replacements <= max_replacements <= holding_count")
    weights = [
        config.strategy.momentum_20_weight,
        config.strategy.momentum_60_weight,
        config.strategy.momentum_120_weight,
        config.strategy.low_volatility_weight,
        config.strategy.low_drawdown_weight,
    ]
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Strategy weights must be non-negative and sum to 1")
    if config.model.target_horizon_days <= 0 or config.model.minimum_training_samples <= 0:
        raise ValueError("Model horizon and minimum training sample count must be positive")
