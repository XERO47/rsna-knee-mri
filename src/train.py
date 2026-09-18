"""Training entry point skeleton.

python -m src.train --config configs/config.yaml
"""
import argparse

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loaded config:", cfg)

    # TODO: build dataset (src.dataset.KneeSeriesDataset), model, and
    # training loop once the train.csv label schema is confirmed.


if __name__ == "__main__":
    main()
