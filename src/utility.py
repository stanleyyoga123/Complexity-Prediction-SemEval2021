import os
import yaml

from src.constant import Path


def get_config(mode):
    mode_map = {"scratch": Path.CONFIG_SCRATCH, "pretrained": Path.CONFIG_PRETRAINED}
    if mode not in mode_map:
        raise ValueError("put mode scratch or pretrained")
    return yaml.safe_load(open(mode_map[mode], "r"))


def get_latest_version(folder, prefix, mode="patch"):
    """
    mode --> (major | minor | patch)
    """
    files = os.listdir(folder)
    versions = []
    for file in files:
        if file.startswith(prefix):
            version = [int(ver) for ver in file.split("-")[-1].split(".")]
            versions.append(version)

    if not versions:
        return "0.0.0"

    major, minor, patch = sorted(versions, reverse=True)[0]
    if mode == "major":
        return f"{major+1}.0.0"
    elif mode == "minor":
        return f"{major}.{minor+1}.0"
    elif mode == "patch":
        return f"{major}.{minor}.{patch+1}"
    else:
        raise ValueError("only support (major | minor | patch)")
