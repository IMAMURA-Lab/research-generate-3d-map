# 物体検出を行うスクリプト
# 独自学習モデル用に未変更

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    train = args.train

    # -----------------------------
    # 設定
    # -----------------------------
    MODEL_PATH = f"..\..\..\ZED\label_studio_project\work\{model_name}/run/runs/detect/{train}\weights/best.pt"  # 学習済みモデルパス
    CONF_THRESH = 0.4
    TARGET_CLASS_NAME = "tape"

    # -----------------------------
    # YOLOv8 モデルロード
    # -----------------------------
    model = YOLO(MODEL_PATH)

    # -----------------------------
    # ZED 初期化
    # -----------------------------
    zed = sl.Camera()
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.PERFORMANCE)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED の初期化に失敗しました")
        exit(1)

    runtime_params = sl.RuntimeParameters()

    image_zed = sl.Mat()
    point_cloud = sl.Mat()

    while True:

        # フレーム取得
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        frame = image_zed.get_data()[:, :, :3].copy()

        # -----------------------------
        # YOLOv8 推論
        # -----------------------------
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)

        for r in results:
            for box in r.boxes:

                # バウンディングボックス
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

                # クラス名
                cls_id = int(box.cls[0])
                class_name = r.names[cls_id]

                if class_name != TARGET_CLASS_NAME:
                    continue

                # 信頼度
                conf = float(box.conf[0])

                # 中心座標
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)

                # -----------------------------
                # 深度取得
                # -----------------------------
                err, point = point_cloud.get_value(cx, cy)
                if err != sl.ERROR_CODE.SUCCESS:
                    continue

                X, Y, Z, _ = point

                distance = np.sqrt(X*X + Y*Y + Z*Z)

                # -----------------------------
                # 描画
                # -----------------------------
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}  dist:{distance:.2f}m"
                cv2.putText(frame, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Viewew [detection_object]", frame)

        if cv2.waitKey(1) == 27:  # ESC キー
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()