"""
config

役割:
    configs/ 以下のyamlファイルを読み込み、辞書として各モジュールに渡す。
"""

from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """指定したyamlファイルを読み込み、辞書として返す。"""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
