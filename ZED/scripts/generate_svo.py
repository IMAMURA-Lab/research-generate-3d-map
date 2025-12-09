# SVOファイル作成用スクリプト
# プログラム停止のコマンドをCtrl-CかEscか検討中
# 基本的な録画機能は動作確認済み
# 小型コンピュータを用いての動作は未確認

import csv  # CSVファイル操作用
import sys  # システム関連（exitなど）用
import pyzed.sl as sl
from signal import signal, SIGINT  # Ctrl+C（SIGINT）を捕まえるためのシグナル処理
import keyboard 
import os  # ファイル・パス操作等
import cv2
import numpy as np
import argparse

# センサー（IMU）データ取得用オブジェクト
sensors_data = sl.SensorsData()

stop_signal = False  # プログラム終了フラグ

# Ctrl+Cでの終了処理を定義
def handler(signum, frame):
    global stop_signal
    stop_signal = True

# Ctrl+C（SIGINT）が押されたときにhandler関数を呼び出す設定
signal(SIGINT, handler)

def main():
    zed = sl.Camera()

    global stop_signal
    global sensors_data

    # -----------------------------------------------------------------------
    # 初期化パラメータの設定
    # -----------------------------------------------------------------------
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15

    # カメラの情報（解像度、焦点距離、センサーサイズなど）が取得可能
    camera_infos = zed.get_camera_information()

    # 画像を格納するための Mat / Pose オブジェクトを生成
    image = sl.Mat()

    # OpenCVウィンドウを作成
    window_name = "Viewer [generate_svo]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ウィンドウ名
    cv2.resizeWindow(window_name, 1000, 800)  # 初期サイズ

    # カメラオブジェクト生成    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        # open に失敗した場合は終了
        print("Camera Open", status, "Exit program.")
        exit(1)

    # -----------------------------------------------------------------------
    # 録画（SVO）設定
    # -----------------------------------------------------------------------
    output_svo_file = opt.output_svo_file  
    svo_path = f"..\..\..\ZED\samples\svo\{output_svo_file}"
    # recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264) # 第一引数：出力ファイルパス、第二引数：圧縮方式（例: H264）
    recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H265) # 第一引数：出力ファイルパス、第二引数：圧縮方式（例: H265）
    err = zed.enable_recording(recording_param)  # 録画を有効にする
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    # grab() 時の実行時パラメータ（例：深度取得の有無、感度設定など）のオブジェクト生成
    runtime_parameters = sl.RuntimeParameters()

    # 録画開始メッセージ（Esc で停止）
    print("Start recording\nStop recording with the 'Esc'")

    frames_recorded = 0  # 録画したフレーム数カウンタ

    # -----------------------------------------------------------------------
    # IMU（センサー）データを CSV に書き出す準備
    # -----------------------------------------------------------------------
    # IMU（加速度・角速度）の値を保存
    # imu_csv_file = open("..\..\..\ZED\samples\imu\imu_data_sample.csv", mode="w", newline="")
    imu_csv_file = open("imu_data_sample.csv", mode="w", newline="")
    csv_writer = csv.writer(imu_csv_file)
    # CSV ヘッダ行（見出し）を書き込む
    csv_writer.writerow(["Frame", "Timestamp (ms)", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"])
    i = 0  # センサー記録用のフレームインデックス

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
            # 左カメラ画像を取得して image に格納
            if zed.retrieve_image(image, sl.VIEW.LEFT) == sl.ERROR_CODE.SUCCESS:

                # ZED の Mat を OpenCV 形式に変換（# NumPy 配列として取得）
                image_ocv = np.array(image.get_data(), dtype=np.uint8, copy=True)

                if image_ocv is None:
                    print("Failed to convert image to numpy array")
                    continue

                # OpenCV ウィンドウに表示
                cv2.imshow(window_name, image_ocv)
            else:
                print("Failed to retrieve image")

            # Escキーが押されたら終了
            if cv2.waitKey(3) & 0xFF == 27:
                print("Stopping recording (esc pressed).")
                break

            frames_recorded += 1
            # '\r' を使って同じ行にフレームカウントを上書き表示（コンソール出力）
            print("Frame count: " + str(frames_recorded), end="\r")

            # タイムスタンプを取得（sl.TIME_REFERENCE.CURRENT: 現在時点の参照）
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # 画像がキャプチャされた時刻

            # センサーデータ（IMU 等）を取得
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                # sensors_data.get_imu_data() で IMU のデータ（加速度・角速度など）を取得
                imu_data = sensors_data.get_imu_data()
                # 加速度ベクトル（線形加速度）：戻り値は (ax, ay, az) のようなタプル/配列
                accel = imu_data.get_linear_acceleration()
                # 角速度（ジャイロ）：(gx, gy, gz)
                gyro = imu_data.get_angular_velocity()
                # 取得した IMU データを CSV に書き込む
                csv_writer.writerow([i, timestamp.get_milliseconds(), accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]])
                i += 1  # CSV に書いた行数（またはサンプル番号）をインクリメント

        # Escキーが押されたら終了
        if keyboard.is_pressed('esc'):
            print("Exiting loop(esc pressed).")
            break

        # # Ctrl+Cが押されたら終了
        # if stop_signal:
        #     print("Exiting loop(Ctrl+C pressed).")
        #     print("Final Frame count: " + str(frames_recorded))
        #     break

    # OpenCV ウィンドウを閉じる
    cv2.destroyAllWindows()

    # 録画を停止
    zed.disable_recording()  

    # 解放処理
    image.free()
    imu_csv_file.flush()
    imu_csv_file.close()
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_svo_file', default= "svo_sample.svo2")
    opt = parser.parse_args()

    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"):
        print("--output_svo_file parameter should be a .svo file but is not : ", opt.output_svo_file, "Exit program.")
        exit()

    main()
