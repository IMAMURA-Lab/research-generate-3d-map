"""
SVORecorder

役割:
    ZEDカメラを起動し、ライブ映像をSVO2ファイルとして記録する。
    現場でセンサーを動かして計測する処理(実機のZEDカメラが必要)。

想定される使い方:
    recorder = SVORecorder(output_path="data/session_001/video.svo2")
    recorder.start()
    ... (現場での計測) ...
    recorder.stop()
"""

from pathlib import Path


class SVORecorder:
    def __init__(self, output_path: str, config: dict | None = None):
        """
        Args:
            output_path: 記録先のSVO2ファイルパス
            config: 解像度・フレームレート・座標系などの設定(configs/capture.yaml由来)
        """
        self.output_path = Path(output_path)
        self.config = config or {}
        # TODO: ZED SDK Camera / InitParameters の初期化

    def start(self) -> None:
        """記録を開始する。"""
        raise NotImplementedError

    def stop(self) -> None:
        """記録を終了し、SVOファイルを確定させる。"""
        raise NotImplementedError
