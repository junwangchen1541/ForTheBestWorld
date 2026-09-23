from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import run_backtest
from .config import load_config
from .data import download_current_universe, load_market_data
from .league import run_league, write_league_outputs
from .strategy import choose_holdings, score_universe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSI 300 weekly rotation research toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download current CSI 300 members")
    download.add_argument("--start", required=True)
    download.add_argument("--end", required=True)
    download.add_argument("--output", default="data/hs300.csv")
    download.add_argument("--universe", choices=("hs300", "all-a"), default="hs300")

    backtest = subparsers.add_parser("backtest", help="Run a historical simulation")
    backtest.add_argument("--config", default="config.toml")
    backtest.add_argument("--data", default="data/hs300.csv")
    backtest.add_argument("--output", default="output")

    league = subparsers.add_parser("league", help="Train and run the six model experiments")
    league.add_argument("--config", default="config.toml")
    league.add_argument("--training-data", default="data/all_a.csv")
    league.add_argument("--evaluation-data", default="data/hs300.csv")
    league.add_argument("--data", help="Deprecated alias for --evaluation-data")
    league.add_argument("--output", default="dashboard/data.js")

    recommend = subparsers.add_parser("recommend", help="Create the latest weekly ranking")
    recommend.add_argument("--config", default="config.toml")
    recommend.add_argument("--data", default="data/hs300.csv")
    recommend.add_argument("--holdings", default="", help="Comma-separated current symbols")
    recommend.add_argument("--output", default="output/latest_ranking.csv")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        destination = download_current_universe(args.start, args.end, args.output, args.universe)
        print(f"Saved market data to {destination}")
        return

    config = load_config(args.config)
    if args.command == "backtest":
        data = load_market_data(args.data)
        result = run_backtest(data, config)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result.equity.to_csv(output / "equity.csv", index=False)
        result.trades.to_csv(output / "trades.csv", index=False)
        (output / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        print(json.dumps(result.metrics, indent=2))
        return
    if args.command == "league":
        evaluation_path = args.data or args.evaluation_data
        training_data = load_market_data(args.training_data)
        evaluation_data = load_market_data(evaluation_path)
        payload = run_league(training_data, evaluation_data, config)
        destination = write_league_outputs(payload, args.output)
        ranking = sorted(
            (
                (strategy["name"], strategy["metrics"]["total_return"])
                for strategy in payload["strategies"]
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        print(f"Six experiment results saved to {destination}")
        for position, (name, total_return) in enumerate(ranking, start=1):
            print(f"{position}. {name}: {total_return:.2%}")
        return

    data = load_market_data(args.data)
    scores = score_universe(data, config.strategy)
    current = {symbol.strip().zfill(6) for symbol in args.holdings.split(",") if symbol.strip()}
    selected, sells, buys = choose_holdings(scores, current, config.portfolio)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination, index=False)
    print(json.dumps({"selected": selected, "sell": sells, "buy": buys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
