# SVOファイル作成用スクリプト
# プログラム停止のコマンドをCtrl-CかEscか検討中
# 基本的な録画機能は動作確認済み
# 小型コンピュータを用いての動作は未確認
# 物体検出や障害物判定は未実装（こちらのスクリプトで編集中）

import csv  # CSVファイル読み書き用モジュール（標準ライブラリ）
import sys  # システム関連（exitなど）用
import pyzed.sl as sl  # ZEDカメラ用のPythonラッパー（pyzed）を sl という短縮名で利用
from signal import signal, SIGINT  # Ctrl+C（SIGINT）を捕まえるためのシグナル処理
import keyboard  # 外部ライブラリ。キーボード入力の状態をポーリングするのに便利（※管理者権限が必要な場合あり）
import argparse  # コマンドライン引数のパーサー
import os  # ファイル・パス操作等
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ハンドラ：Ctrl-C のための終了処理
# ---------------------------------------------------------------------------
# ※注意：Python の signal ハンドラは通常 (signum, frame) の2引数を受け取ります。
# 元の定義 (signal_received, zed, frame) は正しくありません。
# なのでここでは「元のコードを変更せずに注釈を付ける」方針で、
# 正しい定義例をコメントで示します。
#
# 正しいシグナルハンドラの例（推奨）：
# def handler(signal_received, frame):
#     # グローバル変数 zed を参照して記録停止・クローズを行う
#     try:
#         zed.disable_recording()  # 録画を無効化（Recording を止める）
#     except Exception:
#         pass
#     try:
#         zed.close()  # カメラを閉じる（リソース解放）
#     except Exception:
#         pass
#     sys.exit(0)
#
# 元コードのまま（しかしこれはシグネチャが間違っている点に注意）：
# def handler(signal_received,zed, frame):
#     zed.disable_recording()
#     zed.close()
#     sys.exit(0)

# def handler(signal_received, frame):
#     """ SIGINT(Ctrl-C) で呼ばれる。グローバルな zed, imu_csv_file を参照して安全に終了処理を行う。
#     注意：main() 内でこれらをグローバルスコープに割り当てておく必要があります。
#     """
#     global zed, imu_csv_file, recording_enabled
#     try:
#         if 'recording_enabled' in globals() and recording_enabled:
#             try:
#                 zed.disable_recording()
#             except Exception:
#                 pass
#     except Exception:
#         pass
#     try:
#         if 'imu_csv_file' in globals() and imu_csv_file is not None:
#             try:
#                 imu_csv_file.flush()
#                 imu_csv_file.close()
#             except Exception:
#                 pass
#     except Exception:
#         pass
#     try:
#         if 'zed' in globals() and zed is not None:
#             try:
#                 zed.close()
#             except Exception:
#                 pass
#     except Exception:
#         pass
#     sys.exit(0)

# シグナル登録：SIGINT（Ctrl+C）を捕まえて handler を呼ぶ
# ※上で述べた通り、handler の定義は修正すべき（実行時に TypeError になる可能性が高い）
# signal(SIGINT, handler)

def main():
    zed = sl.Camera()
    # センサー（IMUなど）のデータ格納用オブジェクト
    sensors_data = sl.SensorsData()

    # -----------------------------------------------------------------------
    # カメラ初期化パラメータ（InitParameters）について
    # -----------------------------------------------------------------------
    # sl.InitParameters は ZED カメラの初期化設定をまとめたオブジェクトです。
    # （例：解像度、FPS、深度モード、カメラID など）
    init_params = sl.InitParameters()
    # depth_mode を NONE にしている（深度推定を OFF にする設定）
    # (sl.DEPTH_MODE.NONE: 深度マップを生成しないモード)
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    # ここでの設定は zed.open(init_params) を使う場合に反映されます
    # カメラ解像度の指定（sl.RESOLUTION.AUTO はカメラ種別に応じて最適解像度を選択）
    init_params.camera_resolution = sl.RESOLUTION.AUTO  # (解像度の自動選択)
    init_params.camera_fps = 30  # カメラのフレームレート（FPS）を 30 に設定
    # 深度モードを PERFORMANCE に変更（sl.DEPTH_MODE.PERFORMANCE: 高速重視の深度推定）
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    # カメラの情報（解像度、焦点距離、センサーサイズなど）が取得可能
    camera_infos = zed.get_camera_information()

    # 画像を格納するための Mat / Pose オブジェクトを生成
    image = sl.Mat()

    # OpenCVウィンドウを作成
    window_name = "Viewer [generate_svo]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ウィンドウ名
    cv2.resizeWindow(window_name, 1000, 800)  # 初期サイズ

    # -----------------------------------------------------------------------
    # カメラを開く
    # -----------------------------------------------------------------------
    # zed.open(init_params) でカメラを初期化して開く
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        # open に失敗した場合は終了（エラーコードを表示）
        print("Camera Open", status, "Exit program.")
        exit(1)

    # -----------------------------------------------------------------------
    # 録画（SVO）設定
    # -----------------------------------------------------------------------
    # RecordingParameters は SVO ファイルへの録画に関する設定オブジェクトです。
    # 第一引数：出力ファイルパス、第二引数：圧縮方式（例: H264）
    # ※ここで使っている opt 変数は __main__ ブロックで argparse によって生成される
    # グローバル変数 opt を参照しています（関数に引数で渡す方が明示的で望ましい）。
    recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H264)  # H264 圧縮で保存
    err = zed.enable_recording(recording_param)  # 録画を有効にする
    if err != sl.ERROR_CODE.SUCCESS:
        print("Recording ZED : ", err)
        exit(1)

    # runtime_parameters（RuntimeParameters）は、grab() 時の実行時パラメータをまとめるオブジェクトです。
    # （例：深度取得の有無、感度設定など）
    runtime_parameters = sl.RuntimeParameters()

    print("Start recording\nStop recording with the 'Esc'")  # 録画開始メッセージ（Esc で停止）

    frames_recorded = 0  # 録画したフレーム数カウンタ

    # -----------------------------------------------------------------------
    # IMU（センサー）データを CSV に書き出す準備
    # -----------------------------------------------------------------------
    # ここでは "imu_data.csv" を新規に作って IMU（加速度・角速度）の値を保存する
    # ※推奨：with open(...) as imu_csv_file: を使うと例外発生時も自動でクローズされ安全です
    imu_csv_file = open("..\Materials\imu_data_sample.csv", mode="w", newline="")
    csv_writer = csv.writer(imu_csv_file)
    # CSV ヘッダ行（見出し）を書き込む
    csv_writer.writerow(["Frame", "Timestamp (ms)", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"])
    i = 0  # センサー記録用のフレームインデックス

    # メインのループ：grab() でフレームを取得しつつセンサーデータを CSV に書き出す
    while True:
        # grab() はカメラから新しいフレームを取りに行きます。
        # 成功したら sl.ERROR_CODE.SUCCESS が返る（非同期取得やカメラ負荷によって失敗することもある）
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
            # 新しいフレームが正常に取得できたら
            # 左カメラ画像を取得して image に格納
            if zed.retrieve_image(image, sl.VIEW.LEFT) == sl.ERROR_CODE.SUCCESS:
                # ZED の Mat を OpenCV 形式に変換
                image_ocv = image.get_data()  # NumPy 配列として取得
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
            # Timestamp オブジェクトには get_milliseconds() 等のメソッドがあります（ZED SDK の仕様）
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # 画像がキャプチャされた時刻

            # センサーデータ（IMU 等）を取得する
            # get_sensors_data() は sensors_data オブジェクトに現在のセンサー情報を格納します
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                # sensors_data.get_imu_data() で IMU のデータ（加速度・角速度など）を取得
                imu_data = sensors_data.get_imu_data()
                # 加速度ベクトル（線形加速度）：戻り値は (ax, ay, az) のようなタプル/配列
                accel = imu_data.get_linear_acceleration()
                # 角速度（ジャイロ）：(gx, gy, gz)
                gyro = imu_data.get_angular_velocity()
                # 取得した IMU データを CSV に書き込む
                # timestamp.get_milliseconds() はタイムスタンプをミリ秒で返す想定（ZED の Timestamp API）
                csv_writer.writerow([i, timestamp.get_milliseconds(), accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]])
                i += 1  # CSV に書いた行数（またはサンプル番号）をインクリメント

            if frames_recorded % 10000 == 0:  # Print every 10000 frames
                print("Frame count: " + str(frames_recorded), end="\r")

        # keyboard.is_pressed('esc') を使って ESC キーでループを止められるようにしている
        # ※注意：keyboard モジュールは一部環境（特に Linux/Mac/一部 Windows）で管理者権限や特殊な設定が必要
        # またメインループ内でポーリングしているため CPU 負荷が増える可能性あり
        # if keyboard.is_pressed('esc'):
        #     # メッセージの「Stopping recording after 1000 frames.」は固定文であり、
        #     # 実際のフレーム数に依存していない（おそらくハードコーディングされた文言）
        #     print("Stopping recording after 1000 frames.")
        #     break

    # OpenCV ウィンドウを閉じる
    cv2.destroyAllWindows()

    # ループ終了後の解放処理（使ったメモリ・モジュールを閉じる）
    image.free(memory_type=sl.MEM.CPU)  # CPUメモリ上の Mat を解放
    # ループ終了後、CSV ファイルをクローズ（重要：ファイルを確実に保存するため）
    imu_csv_file.close()
    # カメラを閉じる（リソース解放）
    zed.close()
    # 保険的にもう一度メモリを解放しておく（重複しているが安全側の実装）
    image.free()
    # カメラを閉じる（既に閉じている可能性があるが二重呼び出しは安全のため）
    zed.close()


# エントリポイント：コマンドライン引数の受け取り
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --output_svo_file で出力 SVO ファイルのパスを受け取る（必須）
    parser.add_argument('--output_svo_file', type=str, help='Path to the SVO file that will be written', required=True)
    opt = parser.parse_args()

    # 拡張子が .svo または .svo2 でなければエラー表示して終了
    if not opt.output_svo_file.endswith(".svo") and not opt.output_svo_file.endswith(".svo2"):
        print("--output_svo_file parameter should be a .svo file but is not : ", opt.output_svo_file, "Exit program.")
        exit()

    # main を呼ぶ（注意：main の内部でグローバル opt を参照している）
    # 推奨：明示的に main(opt) のように引数で渡す形にして副作用を減らす
    main()
