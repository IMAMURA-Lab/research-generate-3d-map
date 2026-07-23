"""
frame_synchronizer

役割:
    SVOフレームとLiDARフレームを、タイムスタンプの最近傍探索で対応付ける。
    対応表は sync_table.json として書き出し/読み込みができるようにする。

想定される使い方:
    pairs = synchronized_frames(svo_reader, lidar_reader, max_diff_ms=50)
    for svo_frame, lidar_frame in pairs:
        ...
"""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SyncedFrame:
    """時刻同期済みの1組のフレーム。"""

    svo_frame_id: int
    lidar_frame_id: int
    time_diff_ms: float
    rgb_image: object
    camera_pose: object
    lidar_points: object


def synchronized_frames(svo_reader, lidar_reader, max_diff_ms: float = 50.0):
    """
    SVOフレームとLiDARフレームをタイムスタンプで対応付けて順に返す。

    Args:
        svo_reader: SVOReaderのインスタンス
        lidar_reader: LiDARReaderのインスタンス
        max_diff_ms: 対応付けを許容する最大の時刻差(ミリ秒)。
                     これを超える場合はそのフレームを除外する。

    Yields:
        SyncedFrame
    """
    raise NotImplementedError


def save_sync_table(synced_frames: list[SyncedFrame], output_path: str) -> None:
    """対応付け結果(フレームIDのペアと時刻差)をJSONに書き出す。"""
    raise NotImplementedError


def load_sync_table(input_path: str) -> list[dict]:
    """保存済みの対応表を読み込む。"""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)
