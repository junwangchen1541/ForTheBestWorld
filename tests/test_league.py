from dataclasses import replace
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression
from test_strategy import market_data

from hs300_rotation.config import (
    AppConfig,
    BacktestConfig,
    ModelConfig,
    PortfolioConfig,
    StrategyConfig,
)
from hs300_rotation.league import MODEL_PROFILES, POOL_PROFILES, run_league, write_league_outputs
from hs300_rotation.models import (
    MODEL_FEATURES,
    LayaRanker,
    _time_validation,
    _training_frame,
    fit_experiment,
    make_labeled_features,
)


def test_experiment_matrix_is_three_models_by_two_pools() -> None:
    assert [profile.model_id for profile in MODEL_PROFILES] == ["tree", "laya", "mlp"]
    assert POOL_PROFILES == [("all_a", "全 A 股训练"), ("hs300", "沪深300训练")]
    assert len(MODEL_PROFILES) * len(POOL_PROFILES) == 6


def test_demo_dashboard_date_matches_latest_demo_activity() -> None:
    demo_data = Path(__file__).parents[1] / "dashboard" / "demo-data.js"
    source = demo_data.read_text(encoding="utf-8")
    assert '"as_of_date":"2026-09-23"' in source


def test_laya_ranker_uses_official_python_method_and_calibrator() -> None:
    class FakeLaya:
        def __init__(self):
            self.calls = 0

        def system_one(self, state, questions):
            self.calls += 1
            assert "market_features" in state
            assert "expected_return" in questions
            return {"answers": {"expected_return": {"score": 3.0}}}

    agent = FakeLaya()
    calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.0, 4.0], [-0.1, 0.1])
    ranker = LayaRanker(agent, calibrator)
    samples = pd.DataFrame([{name: index / 10 for index, name in enumerate(MODEL_FEATURES)}])
    prediction = ranker.predict(samples)
    assert agent.calls == 1
    assert prediction[0] > 0


def test_training_labels_do_not_cross_cutoff_and_pool_selection() -> None:
    data = market_data(symbols=4, days=140)
    config = ModelConfig(
        training_end_date="2024-06-30",
        target_horizon_days=5,
        minimum_training_samples=1,
    )
    labeled = make_labeled_features(data, config.target_horizon_days)
    hs300_symbols = {"000001", "000002"}
    broad = _training_frame(labeled, "all_a", hs300_symbols, config)
    narrow = _training_frame(labeled, "hs300", hs300_symbols, config)
    assert broad["symbol"].nunique() == 4
    assert set(narrow["symbol"].unique()) == hs300_symbols
    assert (broad["target_date"] <= pd.Timestamp(config.training_end_date)).all()
    assert set(MODEL_FEATURES).issubset(broad.columns)


def test_time_validation_purges_forward_labels() -> None:
    data = market_data(symbols=4, days=160)
    labeled = make_labeled_features(data, horizon=5).dropna(subset=["target_return", *MODEL_FEATURES])
    train, validation = _time_validation(labeled)
    assert not train.empty and not validation.empty
    assert train["target_date"].max() < validation["date"].min()


def test_tree_and_mlp_fit_real_training_samples() -> None:
    data = market_data(symbols=8, days=180)
    model_config = ModelConfig(
        training_end_date=str(sorted(data["date"].unique())[143].date()),
        target_horizon_days=5,
        minimum_training_samples=100,
        laya_calibration_samples=20,
    )
    for model_id in ("tree", "mlp"):
        fitted = fit_experiment(
            training_data=data,
            evaluation_data=data,
            pool="all_a",
            model_id=model_id,
            hs300_symbols=set(data["symbol"].unique()),
            model_config=model_config,
        )
        assert fitted.training_rows >= model_config.minimum_training_samples
        assert len(fitted.score_on(data["date"].max())) == 8


def test_run_league_builds_six_shared_universe_experiments(monkeypatch, tmp_path) -> None:
    data = market_data(symbols=12, days=210)
    data["name"] = "Test " + data["symbol"]
    dates = sorted(data["date"].unique())
    cutoff = pd.Timestamp(dates[135])
    test_start = pd.Timestamp(dates[136])
    config = AppConfig(
        portfolio=PortfolioConfig(
            initial_cash=200_000,
            holding_count=3,
            min_replacements=1,
            max_replacements=1,
        ),
        strategy=replace(StrategyConfig(), minimum_history=121),
        backtest=BacktestConfig(start_date=str(test_start.date()), end_date=str(pd.Timestamp(dates[-1]).date())),
        model=ModelConfig(
            training_end_date=str(cutoff.date()),
            target_horizon_days=5,
            minimum_training_samples=1,
            laya_calibration_samples=20,
        ),
    )

    class DummyEstimator:
        def predict(self, samples):
            return samples["momentum_20"].fillna(0).to_numpy()

    def fake_fit(training_data, evaluation_data, pool, model_id, hs300_symbols, model_config, **kwargs):
        eligible = training_data[training_data.date <= pd.Timestamp(model_config.training_end_date)]
        selected = eligible if pool == "all_a" else eligible[eligible.symbol.isin(hs300_symbols)]
        features = make_labeled_features(evaluation_data, model_config.target_horizon_days)
        features = features.dropna(subset=MODEL_FEATURES)
        return type(
            "DummyExperiment",
            (),
            {
                "model_id": model_id,
                "model_name": model_id,
                "pool_id": pool,
                "pool_name": "all A" if pool == "all_a" else "hs300",
                "estimator": DummyEstimator(),
                "features": features,
                "training_rows": len(selected),
                "training_start": str(selected.date.min().date()),
                "training_end": str(selected.date.max().date()),
                "target_horizon_days": model_config.target_horizon_days,
                "validation_rmse": 0.01,
                "laya_version": None,
                "score_on": lambda self, date: _dummy_scores(self.features, self.estimator, date),
            },
        )()

    monkeypatch.setattr("hs300_rotation.league.fit_experiment", fake_fit)
    payload = run_league(data, data, config)
    assert len(payload["experiments"]) == 6
    assert {item["pool_id"] for item in payload["experiments"]} == {"all_a", "hs300"}
    assert {item["model_id"] for item in payload["experiments"]} == {"tree", "laya", "mlp"}
    assert all(item["id"] != "" for item in payload["experiments"])
    destination = write_league_outputs(payload, tmp_path / "data.js")
    assert destination.read_text(encoding="utf-8").startswith("window.COMPETITION_DATA=")


def _dummy_scores(features, estimator, date):
    cross = features[features["date"] == date].copy()
    cross["score"] = estimator.predict(cross[MODEL_FEATURES])
    cross = cross.sort_values(["score", "symbol"], ascending=[False, True])
    cross["rank"] = range(1, len(cross) + 1)
    return cross
