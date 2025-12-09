# 画像を生成するスクリプト（ZEDカメラからRGB画像と深度マップを取得し保存する）
#  - 実行例: python capture_zed.py --output ./output --res 720
#  - 出力: RGB png, 深度の .npy（32bit float）, 深度可視化用 PNG（8bit 正規化）

import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2
import pyzed.sl as sl

# -----------------------------------------------------------------------------
# ユーティリティ関数
# -----------------------------------------------------------------------------
def save_rgb(image_np, path):
    """
    RGB（正確にはここではBGR）画像をファイルへ保存するラッパー。
    引数:
      - image_np: H x W x 3, dtype=uint8, BGR順（OpenCV 標準）
      - path: 出力ファイルパス（例: "./output/rgb_20250101_120000.png"）
    注意:
      - OpenCV の imwrite は配列が連続（contiguous）であることを期待する場合がある。
    """
    cv2.imwrite(path, image_np)

def save_depth(depth_np, path):
    """
    深度配列（float32, 単位: m）を2形式で保存する:
      1) 生データ: path + ".npy"（numpy の 32-bit float ファイル）
      2) 可視化用: path + "_depth_vis.png"（0..255 に正規化した 8bit PNG）
    深度配列は nan を含むことがある（取得できないピクセル）ため、その扱いに注意する。
    """
    # 1) 生データ保存（.npy）
    np.save(path + ".npy", depth_np)

    # 2) quick-visualization 用に正規化して PNG 保存
    # valid_mask: 有効な深度を持つピクセル（finite かつ正の値）
    valid_mask = np.isfinite(depth_np)
    if valid_mask.any():
        # 表示用の最大距離（メートル） - この値より遠ければ白（最大）でクリップする設計
        max_display_m = 10.0
        disp = depth_np.copy()
        # 無効ピクセルは 0 にする（黒で表示）
        disp[~valid_mask] = 0.0
        # 0..max_display_m の範囲にクリップ
        disp = np.clip(disp, 0.0, max_display_m)
        # 0..max_display_m -> 0..255 にスケールして uint8 に変換
        disp_vis = (disp / max_display_m * 255.0).astype(np.uint8)
    else:
        # 有効な深度が一切ない場合は真っ黒の画像を作る
        disp_vis = np.zeros(depth_np.shape, dtype=np.uint8)

    cv2.imwrite(path + "_depth_vis.png", disp_vis)

# -----------------------------------------------------------------------------
# キャプチャ本体
# -----------------------------------------------------------------------------
def run_capture(args):
    """
    メイン処理: ZED を起動し、キー入力で画像保存／SVO 録画を行う。
    キー:
      - 's': 現在フレームの RGB と depth を保存
      - 'v': SVO 録画の開始/停止（Recorder を使用）
      - 'q' または ESC: 終了
    """

    class_name = args.class_name

    # 出力ディレクトリの作成
    image_dir = f"..\..\..\ZED\label_studio_project\data\classes\{class_name}\images/new"
    depth_dir = f"..\..\..\ZED\label_studio_project\data\classes\{class_name}\depths/new"
    video_dir = f"..\..\..\ZED\label_studio_project\data\classes\{class_name}\videos/new"

    # ----------------------------------------
    # ZED 初期化パラメータ
    # ----------------------------------------
    init = sl.InitParameters()
    # 解像度設定（720 または 1080）
    init.camera_resolution = sl.RESOLUTION.HD720 if args.res == 720 else sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL

    # ZED Camera オブジェクトの作成とオープン
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        # open に失敗したらエラー表示して終了（デバイス未接続やドライバ問題が考えられる）
        print("ZED open failed:", status)
        return

    # pyzed.sl.Mat オブジェクトを用意（画像・深度格納用）
    image = sl.Mat()
    depth = sl.Mat()

    print("ZED opened. Press 's' to save frame, 'v' to toggle SVO recording, 'q' or ESC to quit.")

    recording = False       # 現在録画中かどうかの状態
    svo_filepath = None     # 録画ファイルパス（録画中に設定される）

    try:
        # メインループ
        while True:
            # zed.grab() で新規フレームを取得（内部でセンサ読み取りや同期処理を行う）
            # 成功時は sl.ERROR_CODE.SUCCESS が返る 
            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                # フレーム取得失敗（まれにタイミングで失敗する） -> リトライ
                continue

            # ------------------------------
            # 画像と深度の取得
            # ------------------------------
            # retrieve_image(..., sl.VIEW.LEFT) は左カメラの画像（BGRA U8）を取得する
            zed.retrieve_image(image, sl.VIEW.LEFT)           # BGRA U8 が一般的
            # retrieve_measure(..., sl.MEASURE.DEPTH) は深度（float32, 単位: m）を取得
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)     # float32 の深度マップ

            # pyzed の Mat から numpy 配列を得る（OpenCV で扱える形に整形する）
            img_np = image.get_data()    # 通常 H x W x 4 (BGRA) の ndarray が返る
            if img_np is None:
                # 取得に失敗したら次ループへ
                continue

            # BGRA -> BGR（Alpha チャネルを除去）して連続配列にする
            # np.ascontiguousarray(..., dtype=np.uint8) で OpenCV が扱える形に
            if img_np.ndim == 3 and img_np.shape[2] >= 3:
                # 最低 3 チャネルはある想定（BGRA なら [:,:,:3] で BGR を取り出す）
                frame = np.ascontiguousarray(img_np[:, :, :3].copy(), dtype=np.uint8)
            else:
                # 何か特殊なフォーマットで来た場合はそのまま連続配列化
                frame = np.ascontiguousarray(img_np.copy(), dtype=np.uint8)

            frame_for_saved = frame.copy()  # 保存用にframeをコピー

            # 深度データも numpy 配列として取得し、連続配列にしておく
            depth_np = depth.get_data()  # -> float32 ndarray (H x W)
            # 深度配列は非連続で返る場合があるため明示的にコピーして contiguous にする
            depth_np = np.ascontiguousarray(depth_np.copy(), dtype=np.float32)

            # ------------------------------
            # 画面描画（UI 的な文字列）
            # ------------------------------
            # fps 等を表示したい場合は zed.get_camera_information().camera_fps などを使う
            # ここでは常に表示しないため条件式で False にしている（必要なら True に）
            fps = zed.get_camera_information().camera_fps if False else None
            cv2.putText(frame, f"Press s: save, v: toggle SVO, q: quit", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ウィンドウ表示（OpenCV）
            cv2.imshow("Viewer [generate_image]", frame)
            key = cv2.waitKey(1) & 0xFF

            # ------------------------------
            # キーハンドリング
            # ------------------------------
            if key == ord('s'):
                # 現在フレームを保存（RGB と depth）
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                rgb_path = os.path.join(image_dir, f"image_{ts}.png")
                depth_path_base = os.path.join(depth_dir, f"depth_{ts}")

                # RGB（実際は BGR）が保存される
                save_rgb(frame_for_saved, rgb_path)
                # 深度は .npy（raw float32）と可視化 PNG を保存
                save_depth(depth_np, depth_path_base)
                print(f"Saved: {rgb_path}, {depth_path_base}.npy, {depth_path_base}_depth_vis.png")

            elif key == ord('v'):
                # SVO（ZED の録画フォーマット）録画のトグル（開始／停止）
                # Recorder を使った録画は zed.enable_recording / zed.disable_recording を利用
                if not recording:
                    # 録画を開始する
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    svo_filepath = os.path.join(video_dir, f"record_{ts}.svo")
                    # RecordingParameters の第1引数にファイル名、第2引数に圧縮モード（ここでは Lossless）
                    rec_params = sl.RecordingParameters(svo_filepath, sl.SVO_COMPRESSION_MODE.SVO_COMPRESSION_MODE_LOSSLESS)

                    # enable_recording はエラーコードを返す
                    rc = zed.enable_recording(rec_params)
                    if rc == sl.ERROR_CODE.SUCCESS:
                        recording = True
                        print(f"Started recording SVO -> {svo_filepath}")
                    else:
                        print("Failed to start recording:", rc)
                else:
                    # 録画を停止する
                    zed.disable_recording()
                    recording = False
                    print(f"Stopped recording SVO -> {svo_filepath}")
                    svo_filepath = None

            elif key == ord('q') or key == 27:
                # 'q' または ESC で終了（録画中であれば録画を停止してから抜ける）
                if recording:
                    zed.disable_recording()
                    recording = False
                    print("Stopped recording before exit.")
                break

    finally:
        # 終了処理: OpenCV のウィンドウ破棄と ZED のクローズ（リソース解放）
        cv2.destroyAllWindows()
        zed.close()
        print("ZED closed.")

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="output", help="Output folder")
    parser.add_argument("--res", type=int, default=720, choices=[720, 1080], help="Resolution: 720 or 1080")
    parser.add_argument("--class_name",required=True)
    args = parser.parse_args()

    run_capture(args)

if __name__ == "__main__":
    main()
