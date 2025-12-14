# 物体検出を行うスクリプト
# 実行例
# python detection_object.py
# --model_name: 学習モデル名（label_studio_project/work/以下のフォルダ名）
# --train: 学習モデルが入っているフォルダ名（run/runs/detect/以下のフォルダ名、例：train1）

# 独自学習モデル用に変更済み
# より精密な深度計算を検討中
# 物体を検出した際に、どの段階で座標を登録するのかを検討中
# →今のところは、三秒間検出した時点で登録で検討中
# →1つのモデルに対して、一回までしか登録が出来ないのが問題点
# →物体検出の信頼度を活用する事も検討中

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import argparse

def main():

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    opt = parser.parse_args()
    model_name = opt.model_name
    train = opt.train

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
        "fire": (0, 0, 255), # 赤
        "person": (255, 255, 0) # 黄色
    }

    # -----------------------------------------------------------------------
    # 初期化パラメータの設定
    # -----------------------------------------------------------------------
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER # 単位（メートル）を指定
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # 座標系の選択

    zed = sl.Camera() # カメラオブジェクト生成

    # カメラを開く
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    runtime_params = sl.RuntimeParameters() # Grab() で使うランタイムパラメータ

    # 画像・点群を格納するためのオブジェクト生成
    image = sl.Mat()
    point_cloud = sl.Mat()

    running = True # プログラム実行フラグ
    ZED_running = False # ZEDカメラ動作フラグ

    while running:

        # スペースキー入力で開始
        print("Press 'SPACE' to start", end="\r") # 同じ行に上書き表示
        if cv2.waitKey(1) == 32:
            ZED_running = True

        # メインループ
        while ZED_running:

            # フレーム取得
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # ZED の Mat を OpenCV 形式に変換（# NumPy 配列として取得）
            image_ocv = np.array(image.get_data(), dtype=np.uint8, copy=True)
            image_ocv = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)

            if image_ocv is None:
                print("Failed to convert image to numpy array")
                continue

            # -----------------------------
            # YOLOv8 推論
            # -----------------------------
            results = model.predict(image_ocv, conf=CONF_THRESH, verbose=False)

            for r in results:
                for box in r.boxes:

                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0]) # バウンディングボックス座標

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
                    cv2.rectangle(image_ocv, (xmin, ymin), (xmax, ymax), color, 1) # バウンディングボックス
                    label = f"{class_name} {conf:.2f}  dist:{distance:.2f}m"
                    cv2.putText(image_ocv, label, (xmin, ymin - 10), # 物体名と距離
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

            cv2.imshow("Viewew [detection_object]", image_ocv)

            # ESCキーで終了
            if cv2.waitKey(1) == 27:
                ZED_running = False
                running = False

    # 解放処理
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()