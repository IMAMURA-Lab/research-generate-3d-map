"""ObjectLocalizer のテスト雛形。"""

import pytest


def test_estimate_position_returns_median_of_bbox_points():
    """bbox内の点群の中央値から位置が算出されることを確認する。"""
    pass


def test_estimate_position_returns_none_when_no_points():
    """bbox内に点群が存在しない場合、Noneが返ることを確認する。"""
    pass
