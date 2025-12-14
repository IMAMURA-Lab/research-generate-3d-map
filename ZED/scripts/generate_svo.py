# SVOファイル作成用スクリプト
# 実行例
# python generate_svo.py
# --output_svo_file: 出力SVOファイル名（デフォルト: svo_sample.svo2）

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

sensors_data = sl.SensorsData() # センサー（IMU）データ取得用オブジェクト

stop_signal = False  # プログラム終了フラグ

# Ctrl+Cでの終了処理を定義
def handler(signum, frame):
    global stop_signal
    stop_signal = True

signal(SIGINT, handler) # Ctrl+C（SIGINT）が押されたときにhandler関数を呼び出す設定

def main():

    global stop_signal
    global sensors_data

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_svo_file', default= "svo_sample.svo2")
    opt = parser.parse_args()
    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"):
        print("--output_svo_file parameter should be a .svo file but is not : ", opt.output_svo_file, "Exit program.")
        exit()
    output_svo_file = opt.output_svo_file

    # -----------------------------------------------------------------------
    # 初期化パラメータの設定
    # -----------------------------------------------------------------------
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    # init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15

    zed = sl.Camera() # カメラオブジェクト生成

    # カメラを開く    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        # open に失敗した場合は終了
        print("Camera Open", status, "Exit program.")
        exit(1)

    camera_infos = zed.get_camera_information() # カメラの情報を取得

    image = sl.Mat() # 画像を格納するためのMatオブジェクト生成

    # OpenCVウィンドウを作成
    window_name = "Viewer [generate_svo]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ウィンドウ名
    cv2.resizeWindow(window_name, 1000, 800)  # 初期サイズ

    # -----------------------------------------------------------------------
    # 録画（SVO）設定
    # -----------------------------------------------------------------------
    svo_path = f"..\..\..\ZED\samples\svo\{output_svo_file}"
    # recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264) # 第一引数：出力ファイルパス、第二引数：圧縮方式（例: H264）
    recording_param = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H265) # 第一引数：出力ファイルパス、第二引数：圧縮方式（例: H265）
    err = zed.enable_recording(recording_param)  # 録画を有効にする
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    runtime_parameters = sl.RuntimeParameters() # grab() 時の実行時パラメータのオブジェクト生成

    print("Start recording\nStop recording with the 'Esc'") # 録画開始メッセージ（Esc で停止）

    frames_recorded = 0  # 録画したフレーム数カウンタ

    # -----------------------------------------------------------------------
    # IMU（センサー）データを CSV に書き出す準備
    # -----------------------------------------------------------------------
    # imu_csv_file = open("..\..\..\ZED\samples\imu\imu_data_sample.csv", mode="w", newline="") # IMU（加速度・角速度）の値を保存
    imu_csv_file = open("imu_data_sample.csv", mode="w", newline="") # IMU（加速度・角速度）の値を保存
    csv_writer = csv.writer(imu_csv_file)
    csv_writer.writerow(["Frame", "Timestamp (ms)", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]) # CSV ヘッダ行（見出し）を書き込む
    i = 0  # センサー記録用のフレームインデックス

    running = True # プログラム実行フラグ
    ZED_running = False # ZEDカメラ動作フラグ

    # メインループ
    while running:

        # スペースキー入力で開始
        print("Press 'SPACE' to start", end="\r") # 同じ行に上書き表示
        if keyboard.is_pressed('space'):
            print("Start to program(space key pressed).")
            ZED_running = True
        
        while ZED_running:

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
                
                if zed.retrieve_image(image, sl.VIEW.LEFT) == sl.ERROR_CODE.SUCCESS: # 左カメラ画像を取得して image に格納

                    image_ocv = np.array(image.get_data(), dtype=np.uint8, copy=True) # ZED の Mat を OpenCV 形式に変換（# NumPy 配列として取得）

                    if image_ocv is None:
                        print("Failed to convert image to numpy array")
                        continue

                    cv2.imshow(window_name, image_ocv) # OpenCV ウィンドウに表示
                else:
                    print("Failed to retrieve image")

                # Escキーが押されたら終了
                if cv2.waitKey(3) & 0xFF == 27:
                    print("Stopping recording (esc pressed).")
                    break

                frames_recorded += 1
                
                print("Frame count: " + str(frames_recorded), end="\r") # フレームカウントを上書き表示

                timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # タイムスタンプを取得（画像がキャプチャされた時刻）

                # センサーデータ（IMU 等）を取得
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                    imu_data = sensors_data.get_imu_data() # IMU のデータ（加速度・角速度など）を取得
                    accel = imu_data.get_linear_acceleration() # 加速度ベクトル（線形加速度）：戻り値は (ax, ay, az) のようなタプル/配列
                    gyro = imu_data.get_angular_velocity() # 角速度（ジャイロ）：(gx, gy, gz)
                    csv_writer.writerow([i, timestamp.get_milliseconds(), accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]]) # 取得した IMU データを CSV に書き込む
                    i += 1  # CSV に書いた行数（またはサンプル番号）をインクリメント

            # Escキーが押されたら終了
            if keyboard.is_pressed('esc'):
                print("Exiting loop(esc pressed).")
                ZED_running = False
                running = False

            # Ctrl+Cが押されたら終了
            # if stop_signal:
            #     print("Exiting loop(Ctrl+C pressed).")
            #     print("Final Frame count: " + str(frames_recorded))
            #     ZED_running = False
            #     running = False

    cv2.destroyAllWindows() # OpenCV ウィンドウを閉じる

    zed.disable_recording() # 録画を停止

    # 解放処理
    image.free()
    imu_csv_file.flush()
    imu_csv_file.close()
    zed.close()

if __name__ == "__main__":

    main()
