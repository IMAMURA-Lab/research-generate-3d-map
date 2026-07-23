"""
LiDARReader

役割:
    記録済みのLiDAR点群データを読み込み、フレーム単位で
    点群(points)とタイムスタンプを取り出す。
"""

from pathlib import Path


class LiDARFrame:
    """1フレーム分のLiDAR点群を表す入れ物。"""

    def __init__(self, frame_id: int, timestamp: float, points):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.points = points  # センサー座標系での点群(N x 3 等)


class LiDARReader:
    def __init__(self, lidar_dir: str):
        """
        Args:
            lidar_dir: 記録済みLiDARフレームが入ったディレクトリ
        """
        self.lidar_dir = Path(lidar_dir)
        # TODO: LiDAR機種確定後、ファイル形式に合わせた読み込み処理を実装

    def __iter__(self):
        """フレームを順番に返すイテレータ。"""
        raise NotImplementedError

    def get_frame(self, frame_id: int) -> LiDARFrame:
        """指定フレーム番号の点群を取得する。"""
        raise NotImplementedError
