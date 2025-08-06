# -*- coding: utf-8 -*-
"""
config.yaml 로더
"""
import yaml
import os

def load_config(path: str = "config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
