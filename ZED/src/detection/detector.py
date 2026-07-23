"""
ObjectDetector

役割:
    RGB画像に対してYOLO(Ultralytics)推論を行い、
    要救助者・危険領域などの検出結果(bbox・class・confidence)を返す。
"""

from dataclasses import dataclass


@dataclass
class Detection:
    class_name: str
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float


class ObjectDetector:
    def __init__(self, model_path: str, config: dict | None = None):
        """
        Args:
            model_path: 学習済みYOLOモデル(.pt)のパス
            config: 信頼度しきい値などの設定(configs/detection.yaml由来)
        """
        self.model_path = model_path
        self.config = config or {}
        # TODO: Ultralytics YOLOモデルの読み込み

    def detect(self, rgb_image) -> list[Detection]:
        """RGB画像から検出結果のリストを返す。"""
        raise NotImplementedError
