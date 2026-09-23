from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .strategy import prepare_features

MODEL_FEATURES = [
    "return_5",
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "volatility_20",
    "drawdown_60",
    "amount_log",
    "intraday_range",
]
_LAYA_RAW_SCORE_CACHE: dict[tuple[float, ...], float] = {}


class Ranker(Protocol):
    def predict(self, samples: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FittedExperiment:
    model_id: str
    model_name: str
    pool_id: str
    pool_name: str
    estimator: Ranker
    features: pd.DataFrame
    training_rows: int
    training_start: str
    training_end: str
    target_horizon_days: int
    validation_rmse: float
    laya_version: str | None = None

    def score_on(self, date: pd.Timestamp) -> pd.DataFrame:
        cross_section = self.features[self.features["date"] == date].copy()
        if cross_section.empty:
            return pd.DataFrame(columns=["symbol", "score", "rank"])
        values = self.estimator.predict(cross_section[MODEL_FEATURES])
        cross_section["score"] = np.asarray(values, dtype=float)
        cross_section = cross_section.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        cross_section = cross_section.sort_values(["score", "symbol"], ascending=[False, True])
        cross_section["rank"] = np.arange(1, len(cross_section) + 1)
        return cross_section.reset_index(drop=True)


class LayaRanker:
    """Frozen Laya decision model plus a train-pool-specific score calibrator."""

    def __init__(self, agent: object, calibrator: IsotonicRegression):
        self.agent = agent
        self.calibrator = calibrator

    @staticmethod
    def state_for(row: pd.Series) -> dict[str, object]:
        return {
            "task": "Rank this stock's expected performance against a broad set of Chinese A-share stocks over the next 20 trading sessions.",
            "market_features": {name: round(float(row[name]), 6) for name in MODEL_FEATURES},
        }

    def raw_scores(self, samples: pd.DataFrame) -> np.ndarray:
        questions = {
            "expected_return": {
                "type": "score",
                "instructions": "How strong is this stock's absolute total return over the next 20 trading sessions?",
                "criteria": [
                    "strongly negative total return",
                    "slightly negative or near-zero total return",
                    "modest positive total return",
                    "strong positive total return",
                    "exceptionally strong total return",
                ],
            }
        }
        scores = []
        for _, row in samples.iterrows():
            key = tuple(round(float(row[name]), 6) for name in MODEL_FEATURES)
            cached = _LAYA_RAW_SCORE_CACHE.get(key)
            if cached is not None:
                scores.append(cached)
                continue
            try:
                state = self.state_for(row)
                if hasattr(self.agent, "system_one"):
                    response = self.agent.system_one(state, questions)
                elif hasattr(self.agent, "systemOne"):
                    response = self.agent.systemOne(state, questions)
                else:
                    response = self.agent.predict(state, questions)
                value = response["answers"]["expected_return"]["score"]
            except (KeyError, TypeError, AttributeError) as exc:
                raise RuntimeError("Unexpected Laya SDK response; expected answers.expected_return.score") from exc
            value = float(value)
            scores.append(value)
            if len(_LAYA_RAW_SCORE_CACHE) < 20_000:
                _LAYA_RAW_SCORE_CACHE[key] = value
        return np.asarray(scores, dtype=float)

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        raw = self.raw_scores(samples)
        return self.calibrator.predict(raw)


_LAYA_AGENT: object | None = None


def make_labeled_features(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    features = prepare_features(data)
    grouped = features.groupby("symbol", sort=False)
    features["return_5"] = grouped["close"].pct_change(5, fill_method=None)
    features["amount_log"] = np.log1p(features["amount"].clip(lower=0))
    features["intraday_range"] = (features["close"] - features["open"]) / features["open"]
    forward_price = grouped["close"].shift(-horizon)
    forward_date = grouped["date"].shift(-horizon)
    features["target_return"] = forward_price / features["close"] - 1.0
    features["target_date"] = forward_date
    return features


def _model_factories() -> dict[str, tuple[str, str, object]]:
    return {
        "tree": (
            "梯度提升树",
            "HistGradientBoostingRegressor · 数值特征回归",
            lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    max_iter=120,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    learning_rate=0.06,
                    random_state=17,
                ),
            ),
        ),
        "laya": (
            "Laya + 校准",
            "冻结的 Laya typed-decision 模型 + 股票池专属收益校准",
            None,
        ),
        "mlp": (
            "前馈神经网络",
            "多层感知机回归 · 数值特征基线",
            lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=0.01,
                    learning_rate_init=0.0005,
                    max_iter=1600,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=17,
                ),
            ),
        ),
    }


def _pool_mask(frame: pd.DataFrame, pool: str, hs300_symbols: set[str]) -> pd.Series:
    if pool == "all_a":
        return pd.Series(True, index=frame.index)
    return frame["symbol"].isin(hs300_symbols)


def _training_frame(
    labeled: pd.DataFrame,
    pool: str,
    hs300_symbols: set[str],
    model_config: ModelConfig,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(model_config.training_end_date)
    eligible = labeled[
        (labeled["date"] <= cutoff)
        & (labeled["target_date"] <= cutoff)
        & _pool_mask(labeled, pool, hs300_symbols)
    ].copy()
    eligible = eligible.dropna(subset=["target_return", *MODEL_FEATURES])
    if len(eligible) < model_config.minimum_training_samples:
        raise ValueError(
            f"Training pool {pool!r} has {len(eligible)} usable samples; "
            f"requires {model_config.minimum_training_samples}. "
            "Check data coverage or lower [model].minimum_training_samples."
        )
    return eligible


def _time_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(frame["date"].drop_duplicates()))
    if len(dates) < 2:
        split = max(1, int(len(frame) * 0.8))
        return frame.iloc[:split], frame.iloc[split:]
    validation_start = dates[max(1, int(len(dates) * 0.8))]
    train = frame[
        (frame["date"] < validation_start) & (frame["target_date"] < validation_start)
    ]
    validation = frame[frame["date"] >= validation_start]
    return train, validation


def _fit_laya(train: pd.DataFrame, config: ModelConfig) -> tuple[LayaRanker, float, str]:
    global _LAYA_AGENT
    try:
        import laya
    except ImportError as exc:
        raise RuntimeError(
            "Laya experiment requires the optional dependency. Install it with `pip install -e .[laya]`."
        ) from exc

    if _LAYA_AGENT is None:
        try:
            _LAYA_AGENT = laya.load(
                "convaiinnovations/laya",
                subfolder="multilingual",
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load the official multilingual Laya checkpoint: {exc}") from exc
    agent = _LAYA_AGENT

    train_part, validation = _time_validation(train)
    calibration_count = min(config.laya_calibration_samples, len(train_part))
    calibration_indexes = np.linspace(0, len(train_part) - 1, calibration_count, dtype=int)
    validation_calibration = train_part.iloc[calibration_indexes]
    validation_raw = LayaRanker(
        agent, IsotonicRegression(out_of_bounds="clip")
    ).raw_scores(validation_calibration)
    validation_calibrator = IsotonicRegression(out_of_bounds="clip")
    validation_calibrator.fit(validation_raw, validation_calibration["target_return"].to_numpy(dtype=float))
    if validation.empty:
        rmse = float("nan")
    else:
        raw_valid = LayaRanker(agent, validation_calibrator).raw_scores(validation)
        rmse = float(
            np.sqrt(
                mean_squared_error(
                    validation["target_return"], validation_calibrator.predict(raw_valid)
                )
            )
        )

    calibration_count = min(config.laya_calibration_samples, len(train))
    calibration_indexes = np.linspace(0, len(train) - 1, calibration_count, dtype=int)
    final_calibration = train.iloc[calibration_indexes]
    final_raw = LayaRanker(
        agent, IsotonicRegression(out_of_bounds="clip")
    ).raw_scores(final_calibration)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(final_raw, final_calibration["target_return"].to_numpy(dtype=float))
    version = str(getattr(laya, "__version__", "unknown"))
    return LayaRanker(agent, calibrator), rmse, version


def fit_experiment(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    pool: str,
    model_id: str,
    hs300_symbols: set[str],
    model_config: ModelConfig,
    prepared_training: pd.DataFrame | None = None,
    prepared_evaluation: pd.DataFrame | None = None,
) -> FittedExperiment:
    if model_id not in {"tree", "laya", "mlp"}:
        raise ValueError(f"Unknown model id: {model_id}")
    labeled = prepared_training
    if labeled is None:
        labeled = make_labeled_features(training_data, model_config.target_horizon_days)
    train = _training_frame(labeled, pool, hs300_symbols, model_config)
    train_part, validation = _time_validation(train)

    factories = _model_factories()
    if model_id == "laya":
        estimator, validation_rmse, laya_version = _fit_laya(train, model_config)
        model_name, _thesis, _ = factories[model_id]
    else:
        model_name, _thesis, factory = factories[model_id]
        estimator = factory()
        estimator.fit(train_part[MODEL_FEATURES], train_part["target_return"])
        validation_rmse = (
            float(np.sqrt(mean_squared_error(validation["target_return"], estimator.predict(validation[MODEL_FEATURES]))))
            if not validation.empty
            else float("nan")
        )
        estimator.fit(train[MODEL_FEATURES], train["target_return"])
        laya_version = None

    features = prepared_evaluation
    if features is None:
        features = make_labeled_features(evaluation_data, model_config.target_horizon_days)
    features = features.dropna(subset=MODEL_FEATURES)
    return FittedExperiment(
        model_id=model_id,
        model_name=model_name,
        pool_id=pool,
        pool_name="全 A 股训练" if pool == "all_a" else "沪深300训练",
        estimator=estimator,
        features=features,
        training_rows=len(train),
        training_start=train["date"].min().strftime("%Y-%m-%d"),
        training_end=train["target_date"].max().strftime("%Y-%m-%d"),
        target_horizon_days=model_config.target_horizon_days,
        validation_rmse=validation_rmse,
        laya_version=laya_version,
    )
