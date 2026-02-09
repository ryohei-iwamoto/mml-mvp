import json
import os
import time


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def new_run_id():
    return time.strftime("%Y%m%d_%H%M%S")
