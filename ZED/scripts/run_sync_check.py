"""
SVOとLiDARの同期結果(時刻差の分布等)を確認するデバッグ用スクリプト。
"""

from src.preprocessing.svo_reader import SVOReader
from src.preprocessing.lidar_reader import LiDARReader
from src.preprocessing.frame_synchronizer import synchronized_frames, save_sync_table


def main():
    svo_reader = SVOReader("data/session_001/video.svo2")
    lidar_reader = LiDARReader("data/session_001/lidar/")

    synced = list(synchronized_frames(svo_reader, lidar_reader))

    # TODO: 時刻差の平均・最大値などを表示
    save_sync_table(synced, "data/session_001/sync_table.json")


if __name__ == "__main__":
    main()
