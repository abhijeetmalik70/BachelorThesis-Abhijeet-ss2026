"""
Shared helpers used by every script in the active learning loop.
Keeps path handling and config loading in one place so round-to-round
changes only need to happen in configs/round_config.yaml.
"""
import os
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(config_path: str = None) -> dict:
    config_path = config_path or os.path.join(PROJECT_ROOT, "configs", "round_config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def full_path(relative_path: str) -> str:
    return os.path.join(PROJECT_ROOT, relative_path)


def ensure_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    if id_column not in df.columns:
        df = df.reset_index(drop=True)
        df[id_column] = df.index.astype(str)
    return df


def checkpoint_dir_for_round(cfg: dict, round_num: int = None) -> str:
    round_num = round_num if round_num is not None else cfg["round"]
    d = full_path(os.path.join(cfg["model"]["checkpoint_dir"], f"round_{round_num:02d}"))
    os.makedirs(d, exist_ok=True)
    return d


def round_tag(round_num: int) -> str:
    return f"round_{round_num:02d}"
