"""
CalibrationLoader

役割:
    事前に計算済みのZED<->LiDAR外部キャリブレーション結果(calibration.yaml)を
    読み込み、後段の座標変換処理(transforms.py)へ渡す。

    ※ キャリブレーション自体を「計算する」処理はここには含めない。
       機材構成が変わった際に scripts/run_calibration.py 等で別途算出する想定。
"""

from pathlib import Path
import yaml


class Calibration:
    def __init__(self, calibration_path: str):
        """
        Args:
            calibration_path: calibration.yaml のパス
                (LiDAR->ZED間の回転・並進などが記載されている想定)
        """
        self.calibration_path = Path(calibration_path)
        self.rotation = None      # TODO: yamlから読み込み
        self.translation = None   # TODO: yamlから読み込み
        self._load()

    def _load(self) -> None:
        raise NotImplementedError

    def get_transform(self):
        """LiDAR座標系 -> ZED(ワールド)座標系への変換を返す。"""
        raise NotImplementedError
