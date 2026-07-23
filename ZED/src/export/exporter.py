"""
exporter

役割:
    生成したメッシュと物体リストを、Unity側(先輩のUnityプロジェクト)が
    読み込める形式で出力する。
"""

from pathlib import Path
import json


def save_mesh(mesh, output_path: str) -> None:
    """メッシュを.obj形式で書き出す。"""
    raise NotImplementedError


def save_objects(objects, output_path: str) -> None:
    """
    物体リスト(id・class・position等)をJSON形式で書き出す。

    Args:
        objects: Tracker3D.get_final_objects() の結果
        output_path: 出力先(例: data/map/objects.json)
    """
    raise NotImplementedError
