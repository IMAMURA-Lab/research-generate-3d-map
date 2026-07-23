"""
物体検出処理だけを単体で試すためのスクリプト(SVOのRGBフレームに対して実行)。
"""

from src.preprocessing.svo_reader import SVOReader
from src.detection.detector import ObjectDetector


def main():
    svo_reader = SVOReader("data/session_001/video.svo2")
    detector = ObjectDetector(model_path="models/best.pt")

    for frame in svo_reader:
        detections = detector.detect(frame.rgb_image)
        print(frame.frame_id, detections)


if __name__ == "__main__":
    main()
