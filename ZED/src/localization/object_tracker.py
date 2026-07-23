"""
Tracker3D

役割:
    3D座標ベースのTracking-by-Detectionにより、フレームをまたいで
    物体にIDを付与・更新する(MOT: Multiple Object Tracking)。
    これにより「1クラス1物体」までしか扱えなかった制限を解消する。
"""

from dataclasses import dataclass


@dataclass
class TrackedObject:
    object_id: int
    class_name: str
    position: tuple  # (x, y, z)
    last_seen_frame: int
    update_count: int


class Tracker3D:
    def __init__(self, distance_threshold: float = 0.5, config: dict | None = None):
        """
        Args:
            distance_threshold: 同一物体とみなす距離のしきい値(メートル)
                                 ※ まず固定値で実験し、必要ならクラス別に調整する
            config: 更新方式(上書き/平均・フィルタ)などの設定
        """
        self.distance_threshold = distance_threshold
        self.config = config or {}
        self._next_id = 0
        self._tracked_objects: dict[int, TrackedObject] = {}

    def update(self, measurements: list[dict]) -> None:
        """
        1フレーム分の検出結果(class_name, position等を含む)を受け取り、
        既存IDとの距離マッチングを行い、IDの更新/新規発行を行う。
        """
        raise NotImplementedError

    def get_final_objects(self) -> list[TrackedObject]:
        """全フレーム処理後の最終的な物体リストを返す。"""
        raise NotImplementedError
