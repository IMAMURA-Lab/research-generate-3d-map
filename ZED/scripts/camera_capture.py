# SVOファイルを生成するためのスクリプト
# 実行例
# python camera_capture.py
# --output_svo_file : 出力するSVOファイル名（デフォルト：svo_sample.svo2）

# ビューワなし
# 処理を軽くできるよう調整中

import csv  # CSVファイルへの書き込み用ライブラリ
import sys  # システム関連機能の利用（終了など）
import pyzed.sl as sl
from signal import signal, SIGINT  # Ctrl+Cなどのシグナルを扱うため
import argparse
import os  # OS関連操作用

sensors_data = sl.SensorsData() # センサー（IMU）データ取得用オブジェクト

stop_signal = False # プログラム終了フラグ

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
    parser.add_argument('--output_svo_file', default="svo_sample.svo2")
    opt = parser.parse_args()
    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"): 
        print("--output_svo_file parameter should be a .svo file but is not : ",opt.output_svo_file,"Exit program.")
        exit()
    file_name = opt.output_svo_file

    zed = sl.Camera() # カメラオブジェクト生成

    # -----------------------------------------------------------------------
    # 初期化パラメータの設定
    # -----------------------------------------------------------------------
    init_params = sl.InitParameters()
    # init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
   
    # カメラを開く
    status = zed.open(init_params) 
    if status != sl.ERROR_CODE.SUCCESS: 
        print("Camera Open", status, "Exit program.")
        exit(1)

    # -----------------------------------------------------------------------
    # 録画（SVO）設定
    # -----------------------------------------------------------------------
    # recording_param = sl.RecordingParameters(file_name, sl.SVO_COMPRESSION_MODE.H264)
    recording_param = sl.RecordingParameters(file_name, sl.SVO_COMPRESSION_MODE.H265)
    err = zed.enable_recording(recording_param)  # 録画を有効化
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    runtime_parameters = sl.RuntimeParameters() # grab() 時の実行時パラメータのオブジェクト生成
    
    frames_recorded = 0  # 録画フレーム数カウンタ

    # IMUデータを記録するCSVファイルを作成
    imu_csv_file = open("imu_data.csv", mode="w", newline="")
    csv_writer = csv.writer(imu_csv_file)
    csv_writer.writerow(["Frame", "Timestamp (ms)", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]) # CSVのヘッダー行

    i = 0  # IMUデータのフレーム番号カウンタ

    # メインループ
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            frames_recorded += 1

            print("Frame count: " + str(frames_recorded), end="\r") # 進捗表示（同じ行に上書き）

            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT) # 画像取得時のタイムスタンプを取得（ミリ秒単位）

            # センサー（IMU）データを取得
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                imu_data = sensors_data.get_imu_data()  # IMUデータを取得
                accel = imu_data.get_linear_acceleration()  # 加速度
                gyro = imu_data.get_angular_velocity()     # 角速度

            # CSVにIMUデータを書き込み
            csv_writer.writerow([i, timestamp.get_milliseconds(),
                                     accel[0], accel[1], accel[2],
                                     gyro[0], gyro[1], gyro[2]])

            i += 1  # 次のフレーム番号に進む

        # 録画フレーム数が100万に達したら停止
        if frames_recorded >= 1000000:
            print("Stopping recording after 1000000 frames.")
            print("Final Frame count: " + str(frames_recorded))
            break

        # Ctrl+Cが押されたら停止
        if stop_signal:
            print("Exiting loop(Ctrl+C pressed).")
            print("Final Frame count: " + str(frames_recorded))
            break

    # CSVファイルを閉じる
    imu_csv_file.flush()
    imu_csv_file.close()

    zed.disable_recording() # 録画を停止
    zed.close() # カメラを閉じる
    sys.exit(0) # プログラムを終了

if __name__ == "__main__":

    main()
