"""
ObjectLocalizer

役割:
    検出されたbbox内の領域に対応するLiDAR点群を抽出し、
    中央値等から物体の3Dワールド座標を決定する。
"""


class ObjectLocalizer:
    def __init__(self, calibration, config: dict | None = None):
        """
        Args:
            calibration: Calibration インスタンス
            config: 位置推定の設定(configs/detection.yaml等)
        """
        self.calibration = calibration
        self.config = config or {}

    def estimate_position(self, bbox, lidar_points, camera_pose):
        """
        Args:
            bbox: 検出結果のバウンディングボックス(x1, y1, x2, y2)
            lidar_points: 同時刻のLiDAR点群(ワールド座標変換済み想定)
            camera_pose: 同時刻のカメラ姿勢

        Returns:
            物体の3Dワールド座標(x, y, z)。推定できない場合は None。
        """
        raise NotImplementedError
