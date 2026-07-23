"""
Entry point for running the world model simulation.
Usage: python sim/run.py --config configs/baseline_sp500.yaml
"""
import argparse
import yaml

from world.engine import WorldEngine
from world.clock import MarketClock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_sp500.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    clock = MarketClock(
        start=cfg["simulation"]["start_date"],
        freq=cfg["simulation"]["tick_freq"],
    )

    # TODO: instantiate agents from config
    agents = []
    tickers = []  # TODO: load from universe

    engine = WorldEngine(agents=agents, tickers=tickers, clock=clock)
    results = engine.run(n_steps=cfg["simulation"]["n_steps"])

    print(f"Simulation complete. {len(results)} steps.")


if __name__ == "__main__":
    main()
