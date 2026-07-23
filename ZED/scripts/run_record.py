"""
現場でZED/LiDARを起動し、計測(記録)を行う実行スクリプト。
"""

from src.capture.svo_recorder import SVORecorder
from src.capture.lidar_recorder import LiDARRecorder
from src.common.config import load_config


def main():
    config = load_config("configs/capture.yaml")

    svo_recorder = SVORecorder(
        output_path="data/session_001/video.svo2", config=config
    )
    lidar_recorder = LiDARRecorder(
        output_dir="data/session_001/lidar/", config=config
    )

    svo_recorder.start()
    lidar_recorder.start()

    input("計測中... 終了するにはEnterを押してください\n")

    svo_recorder.stop()
    lidar_recorder.stop()


if __name__ == "__main__":
    main()
