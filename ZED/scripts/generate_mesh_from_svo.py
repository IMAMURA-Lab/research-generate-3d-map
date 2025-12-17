# SVOファイルまたはSVO2ファイルからメッシュを生成するスクリプト
# 実行例
# python generate_mesh_from_svo.py
# --svo_dir: SVOファイルが保存されているディレクトリ（デフォルト: samples）
# --input_svo_file: 入力SVOファイル名（デフォルト: svo_sample.svo2）
# --mesh_dir: 生成されたメッシュファイルを保存するディレクトリ（デフォルト: samples）
# --output_mesh_file: 出力メッシュファイル名（デフォルト: mesh_sample.obj）
# --frame_step: 処理するフレームの間隔（デフォルト: 1、例: 2なら1フレームおきに処理）

# 基本的な空間マッピング機能は動作確認済み
# メッシュファイルが最後まで保存できない不具合あり
# 処理を軽くできるよう調整中
# →処理するフレーム数の調整
# →パラメータの調整
# →チャンクの更新頻度の調整
# →メッシュ抽出の頻度の調整
# →一度最低限のパフォーマンスを試す方向で検討中
# 総フレーム数に応じて、処理するフレーム数、チャンクの更新頻度、メッシュ抽出の頻度が変更される仕様を検討中
# 

import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse
import keyboard

def main():

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo_dir', default= "samples")
    parser.add_argument('--input_svo_file', default= "svo_sample.svo2")
    parser.add_argument('--mesh_dir', default= "samples")
    parser.add_argument('--output_mesh_file', default= "mesh_sample.obj")
    parser.add_argument('--frame_step', type=int, default=1)
    # parser.add_argument('--total_of_frames', type=int, required=True)
    opt = parser.parse_args()
    svo_dir = opt.svo_dir
    input_svo_file = opt.input_svo_file
    mesh_dir = opt.mesh_dir
    output_mesh_file = opt.output_mesh_file
    frame_step = opt.frame_step # フレーム間隔
    # total_of_frames = opt.total_of_frames

    # -----------------------------------------------------------------------
    # 初期化パラメータの設定
    # -----------------------------------------------------------------------
    init_params = sl.InitParameters()
    parse_args(init_params, svo_dir, input_svo_file)
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL # 深度モードの設定
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY # 深度モードの設定
    init_params.coordinate_units = sl.UNIT.METER # 単位（メートル）を指定
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # 座標系の選択
    # init_params.depth_maximum_distance = 8.0 # 深度（距離）を何メートルまで有効とするか（max distance）
    init_params.depth_maximum_distance = 6.0 # 深度（距離）を何メートルまで有効とするか（max distance）

    zed = sl.Camera() # カメラオブジェクト生成

    # カメラを開く
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit(1)

    # トラッキング／マッピングの状態変数（初期はオフ）
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    camera_infos = zed.get_camera_information() # カメラの情報を取得

    # Grab() で使うランタイムパラメータ
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    # 画像・ポーズを格納するためのオブジェクト生成
    image = sl.Mat()
    pose = sl.Pose()

    # ---------------------------------------------
    # Positional Trackingの設定・有効化
    # ---------------------------------------------
    positional_tracking_params = sl.PositionalTrackingParameters()
    positional_tracking_params.enable_area_memory = True  # カメラ移動に応じたマップ構築を許可
    positional_tracking_params.set_floor_as_origin = False # 原点設定
    positional_tracking_params.enable_imu_fusion = True     # IMUを統合
    err = zed.enable_positional_tracking(positional_tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking:", err)
        zed.close()
        exit(1)
    positional_tracking_params.set_floor_as_origin = True # 床を原点にする

    # 位置推定の基準（原点）をリセット
    init_params_pose = sl.Transform()
    zed.reset_positional_tracking(init_params_pose) 

    # ---------------------------------------------
    # 空間マッピングの設定・有効化
    # ---------------------------------------------
    spatial_mapping_params = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        mapping_range=sl.MAPPING_RANGE.MEDIUM,
        max_memory_usage=2048,
        save_texture=True,
        use_chunk_only=True,
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.MESH
    )
    spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    pymesh = sl.Mesh()
    returned_state = zed.enable_spatial_mapping(spatial_mapping_params)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Spatial Mapping : " + repr(returned_state) + ". Exit program.")
        exit(1)

    # OpenGL ビューワ（ogl_viewer）を生成
    viewer = gl.GLViewer()
    pymesh.clear()
    viewer.clear_current_mesh()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, pymesh, 1)

    end_of_file = False # SVOの終端に到達したかどうかのフラグ

    frame_count = 0  # 処理したフレーム数カウンタ
    extract_mesh_frame_count = 0 # メッシュを抽出する時用のフレーム数カウンタ
    update_chunk_count = 0 # チャンクを更新する時用のフレーム数カウンタ

    running = True # プログラム実行フラグ
    zed_running = False # ZEDカメラ動作フラグ
    mapping_running = False # 空間マッピング実行フラグ

    while running:

        # スペースキー入力で開始
        print("Press 'SPACE' to start", end="\r") # 同じ行に上書き表示
        if keyboard.is_pressed('space'):
            print("Start to program(space key pressed).")
            zed_running = True

        # メインループ
        while viewer.is_available() and zed_running:
        
            err = zed.grab(runtime_params) # 新しいフレームを取得

            frame_count += 1
            print(f"frame_count: {frame_count}", end="\r")

            # print(f"frame/total_of_frames : {frame_count}/{total_of_frames}", end="\r")

            # 指定されたフレーム間隔で処理をスキップ
            if frame_count % frame_step != 0:
                continue 

            if err == sl.ERROR_CODE.SUCCESS:

                if mapping_running == False:
                    mapping_running = True

            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
 
                print("End of SVO reached.")
                end_of_file = True
                if mapping_running == False:
                    mapping_running = True

            else:
                print("Failed to get the frame.")
                mapping_running = False
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)

            if mapping_running:

                tracking_state = zed.get_position(pose)
                mapping_state = zed.get_spatial_mapping_state()

                if update_chunk_count >= 3:
                    zed.request_spatial_map_async() # 更新されたチャンクを非同期で作成するよう要求
                    if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                        zed.retrieve_spatial_map_async(pymesh) # GPU側で処理が終わった最新チャンクをpymesh.chunksにコピー（部分更新）
                        viewer.update_chunks() # 外部からチャンクが更新されたことを通知
                        print("Complete Update chunk.\n")
                    update_chunk_count = 0

                extract_mesh_frame_count += 1
                print(f"extract_mesh_frame_count: {extract_mesh_frame_count}", end="\r")
                if extract_mesh_frame_count >= 10:
                    zed.extract_whole_spatial_map(pymesh) # pymeshに含まれる全チャンクを結合し、抽出
                    print("Complete extract mesh.\n")
                    extract_mesh_frame_count = 0
            
                viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

            if end_of_file:
                zed_running = False
                running = False

            if keyboard.is_pressed('esc'):
                print("Exiting loop(esc key pressed).")
                zed_running = False
                running = False

    # 解放処理1
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()

    viewer.clear_current_mesh()   

    zed.extract_whole_spatial_map(pymesh) # pymeshに含まれる全チャンクを結合し、抽出
    print("Complete extract mesh.\n")

    # メッシュフィルタリング
    filter_params = sl.MeshFilterParameters()
    filter_params.set(sl.MESH_FILTER.MEDIUM)
    pymesh.filter(filter_params, True)

    if spatial_mapping_params.save_texture:
        print("Save texture set to : {}".format(spatial_mapping_params.save_texture))
        pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

    mesh_path = f"..\..\..\ZED\{mesh_dir}\mesh\{output_mesh_file}"
    
    status = pymesh.save(mesh_path)
    if status:
        print("Mesh saved under " + mesh_path)
    else:
        print("Failed to save the mesh under " + mesh_path)

    # 解放処理2
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED
    pymesh.clear()
    image.free()
    zed.close()

def parse_args(init_params, svo_dir, input_svo_file):

    svo_path = f"..\..\..\ZED\{svo_dir}\svo\{input_svo_file}"
    
    if len(svo_path) > 0 and (svo_path.endswith(".svo") or svo_path.endswith(".svo2")):
        init_params.set_from_svo_file(svo_path)
        print("[Sample] Using SVO File input: {0}".format(svo_path))

if __name__ == "__main__":

    main()

