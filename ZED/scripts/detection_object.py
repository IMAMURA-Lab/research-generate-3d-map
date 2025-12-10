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
    # YOLOv8モデル設定
    # -----------------------------
    MODEL_PATH = f"..\..\..\ZED\label_studio_project\work\{model_name}/run/runs/detect/{train}\weights/best.pt"  # 学習済みモデルパス
    CONF_THRESH = 0.4
    TARGET_CLASS_NAMES = ["fire", "person"]  # 検出対象クラス名のリスト
    model = YOLO(MODEL_PATH)

    # ----------------------------------------
    # クラスごとに色を定義（任意に追加）
    # ----------------------------------------
    CLASS_COLORS = {
        "fire": (0, 0, 255),         # 赤
        "person": (255, 255, 0)      # 黄色
    }

    # -----------------------------
    # ZED 初期化
    # -----------------------------
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    status = zed.open(init_params)
    
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    runtime_params = sl.RuntimeParameters()

    image = sl.Mat()
    point_cloud = sl.Mat()

    while True:

        # フレーム取得
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # ZED の Mat を OpenCV 形式に変換（# NumPy 配列として取得）
        image_ocv = np.array(image.get_data(), dtype=np.uint8, copy=True)

        if image_ocv is None:
            print("Failed to convert image to numpy array")
            continue

        # -----------------------------
        # YOLOv8 推論
        # -----------------------------
        results = model.predict(image_ocv, conf=CONF_THRESH, verbose=False)

        for r in results:
            for box in r.boxes:

                # バウンディングボックス
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

                # クラス名
                cls_id = int(box.cls[0])
                class_name = r.names[cls_id]

                if class_name not in TARGET_CLASS_NAMES:
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

                color = CLASS_COLORS.get(class_name, (255, 255, 255))  # デフォルトは白

                cv2.rectangle(image_ocv, (xmin, ymin), (xmax, ymax), color, 1)
                label = f"{class_name} {conf:.2f}  dist:{distance:.2f}m"
                cv2.putText(image_ocv, label, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

        cv2.imshow("Viewew [detection_object]", image_ocv)

        if cv2.waitKey(1) == 27:  # ESC キー
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()