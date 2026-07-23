"""
LiDARRecorder

役割:
    LiDARを起動し、点群データを記録する。
    現場でセンサーを動かして計測する処理(実機のLiDARが必要)。
    ※ LiDAR機種選定後、対応SDK/ドライバに合わせて実装する。

想定される使い方:
    recorder = LiDARRecorder(output_dir="data/session_001/lidar/")
    recorder.start()
    ... (現場での計測) ...
    recorder.stop()
"""

from pathlib import Path


class LiDARRecorder:
    def __init__(self, output_dir: str, config: dict | None = None):
        """
        Args:
            output_dir: 点群フレームを保存するディレクトリ
            config: LiDAR固有の設定(configs/capture.yaml由来)
        """
        self.output_dir = Path(output_dir)
        self.config = config or {}
        # TODO: LiDAR機種確定後、SDK/ドライバの初期化処理を実装

    def start(self) -> None:
        """記録を開始する。"""
        raise NotImplementedError

    def stop(self) -> None:
        """記録を終了する。"""
        raise NotImplementedError
