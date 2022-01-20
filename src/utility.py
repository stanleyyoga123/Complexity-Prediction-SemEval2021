import yaml

from src.constant import Path


def get_config(mode):
    mode_map = {"scratch": Path.CONFIG_SCRATCH, "pretrained": Path.CONFIG_PRETRAINED}
    if mode not in mode_map:
        raise ValueError("put mode scratch or pretrained")
    return yaml.safe_load(open(mode_map[mode], "r"))
