"""
SVOReader

役割:
    記録済みのSVO2ファイルを読み込み、フレーム単位でRGB画像・
    カメラ姿勢(camera_pose)・タイムスタンプを取り出す。
    実機のZEDカメラは不要(Playback modeでの読み込み)。
"""

from pathlib import Path


class SVOFrame:
    """1フレーム分のSVOデータを表す入れ物。"""

    def __init__(self, frame_id: int, timestamp: float, rgb_image, camera_pose):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.rgb_image = rgb_image
        self.camera_pose = camera_pose


class SVOReader:
    def __init__(self, svo_path: str):
        """
        Args:
            svo_path: 読み込むSVO2ファイルのパス
        """
        self.svo_path = Path(svo_path)
        # TODO: ZED SDK Camera(Playback mode) の初期化

    def __iter__(self):
        """フレームを順番に返すイテレータ。"""
        raise NotImplementedError

    def get_frame(self, frame_id: int) -> SVOFrame:
        """指定フレーム番号のデータを取得する。"""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
