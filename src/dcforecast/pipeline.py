"""Command-line entry point.

    python -m dcforecast.pipeline            # bronze -> silver -> gold -> train
    python -m dcforecast.pipeline --stage silver
"""

from __future__ import annotations

import argparse
import logging
import time

from . import bronze, config, gold, silver, train

STAGES = ["bronze", "silver", "gold", "train"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Data centre load forecasting pipeline")
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all",
                        help="Run one stage, or 'all' (default) to run every stage in order")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    config.ensure_dirs()
    runners = {"bronze": bronze.ingest, "silver": silver.build, "gold": gold.build, "train": train.run}

    for stage in STAGES if args.stage == "all" else [args.stage]:
        t0 = time.perf_counter()
        runners[stage]()
        logging.info("Stage %-6s done in %.1fs", stage, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
