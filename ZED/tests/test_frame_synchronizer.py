"""frame_synchronizer のテスト雛形。"""

import pytest


def test_synchronized_frames_matches_closest_timestamp():
    """タイムスタンプが最も近いフレーム同士が対応付けられることを確認する。"""
    # TODO: ダミーのSVOReader/LiDARReaderを用意してテストする
    pass


def test_synchronized_frames_excludes_pairs_over_threshold():
    """しきい値を超える時刻差のペアが除外されることを確認する。"""
    pass
