# 物体検出を行うスクリプト
# 実行例
# python detection_object.py
# --model_name: 学習モデル名（label_studio_project/work/以下のフォルダ名）
# --train: 学習モデルが入っているフォルダ名（run/runs/detect/以下のフォルダ名、例：train1）
# --step: 深度計算をする際の計算する間隔（デフォルト 5）

# 独自学習モデル用に変更済み
# より精密な深度計算に変更済み（未検証）
# 物体を検出した際に、どの段階で座標を登録するのかを検討中
# →今のところは、三秒間検出した時点で登録で検討中
# →1つのモデルに対して、一回までしか登録が出来ないのが問題点
# →物体検出の信頼度を活用する事も検討中

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
import argparse
import keyboard

def distance_calculation(depth, xmax, xmin, ymax, ymin, step):

    height = ymax - ymin
    width = xmax - xmin

    if width <= 0 or height <= 0:
        return None
    
    samples = []

    for y in range(ymin, ymax+1, step):
        for x in range(xmin, xmax+1, step):

            err, dpt = depth.get_value(x, y)

            # 無効値チェック
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            if dpt is None:
                continue
            try:
                dpt_value = float(dpt)
            except:
                continue
            if dpt_value == 0.0 or np.isinf(dpt_value) or np.isnan(dpt_value):
                continue

            samples.append(dpt_value)

    if len(samples) == 0:
        return None
    
    return float(np.median(np.array(samples)))

def main():

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--step", type=int, default=5)
    opt = parser.parse_args()
    model_name = opt.model_name
    train = opt.train
    step = opt.step

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
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.METER # 単位（メートル）を指定
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # 座標系の選択

    zed = sl.Camera() # カメラオブジェクト生成

    # カメラを開く
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    # トラッキングの状態変数（初期はオフ）
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF

    # ---------------------------------------------
    # Positional Tracking の有効化
    # ---------------------------------------------
    # positional_tracking_params = sl.PositionalTrackingParameters()
    # positional_tracking_params.enable_area_memory = True  # カメラ移動に応じたマップ構築を許可
    # positional_tracking_params.enable_imu_fusion = True     # IMUを統合
    # positional_tracking_params.set_floor_as_origin = True  # 原点設定
    # err = zed.enable_positional_tracking(positional_tracking_params)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     print("Failed to enable positional tracking:", err)
    #     zed.close()
    #     exit(1)

    # positional_tracking_params.set_floor_as_origin = True # 床を原点にする

    runtime_params = sl.RuntimeParameters() # Grab() で使うランタイムパラメータ

    # 画像・深度・自己位置を格納するためのオブジェクト生成
    image = sl.Mat()
    # point_cloud = sl.Mat()
    depth = sl.Mat()
    pose = sl.Pose()

    running = True # プログラム実行フラグ
    zed_running = False # ZEDカメラ動作フラグ

    while running:

        # スペースキー入力で開始
        print("Press 'SPACE' to start", end="\r") # 同じ行に上書き表示
        if keyboard.is_pressed('space'):
            print("Start to program(space key pressed).")
            zed_running = True

        # メインループ
        while zed_running:

            # フレーム取得
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            # zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

            zed.retrieve_image(image, sl.VIEW.LEFT)
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

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
                for bbox in r.boxes:

                    xmin, ymin, xmax, ymax = map(int, bbox.xyxy[0]) # バウンディングボックス座標

                    # クラス名
                    cls_id = int(bbox.cls[0])
                    class_name = r.names[cls_id]

                    if class_name not in TARGET_CLASS_NAMES:
                        continue

                    # 信頼度
                    conf = float(bbox.conf[0])

                    distance = distance_calculation(depth, xmax, xmin, ymax, ymin, step)

                    # -----------------------------
                    # 描画
                    # -----------------------------
                    color = CLASS_COLORS.get(class_name, (255, 255, 255))  # デフォルトは白
                    cv2.rectangle(image_ocv, (xmin, ymin), (xmax, ymax), color, 1) # バウンディングボックス

                    if distance is None:
                        label = f"{class_name} {conf:.2f}  dist: error"
                    else:
                        label = f"{class_name} {conf:.2f}  dist:{distance:.2f}m"

                    cv2.putText(image_ocv, label, (xmin, ymin - 10), # 物体名と距離
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Viewew [detection_object]", image_ocv)
            cv2.waitKey(1)

            if keyboard.is_pressed('esc'):
                print("Exiting loop(esc key pressed).")
                zed_running = False
                running = False

    # 解放処理
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    main()