# SVOファイルまたはSVO2ファイルからメッシュを生成するスクリプト
# 実行例
# python generate_mesh_from_svo.py
# --svo_dir: SVOファイルが保存されているディレクトリ（デフォルト: samples）
# --input_svo_file: 入力SVOファイル名（デフォルト: svo_sample.svo2）
# --mesh_dir: 生成されたメッシュファイルを保存するディレクトリ（デフォルト: samples）
# --output_mesh_file: 出力メッシュファイル名（デフォルト: mesh_sample.obj）
# --speed: SVO再生速度の調整パラメータ（デフォルト: 1.0）
# --frame_step: 処理するフレームの間隔（デフォルト: 1、例: 2なら1フレームおきに処理）

# 基本的な空間マッピング機能は動作確認済み
# 処理を軽くできるよう調整中
# メッシュのテクスチャが最後まで保存できない不具合あり
# SVOの再生速度を調整する方法を検討中
# →処理するフレーム数を減らす

import sys
import time
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import argparse
import keyboard

def main():

    # --- 初期化パラメータの設定 ---
    init_params = sl.InitParameters()
    parse_args(init_params)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # 深度モードの設定
    init_params.coordinate_units = sl.UNIT.METER # 単位（メートル）を指定
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # 座標系の選択
    init_params.depth_maximum_distance = 8.0 # 深度（距離）を何メートルまで有効とするか（max distance）

    # カメラオブジェクト生成
    zed = sl.Camera()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()

    # トラッキング／マッピングの状態変数（初期はオフ）
    tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # カメラの情報（解像度、焦点距離、センサーサイズなど）を取得
    camera_infos = zed.get_camera_information()

    # Positional Tracking の有効化
    positional_tracking_params = sl.PositionalTrackingParameters()
    positional_tracking_params.enable_area_memory = True  # カメラ移動に応じたマップ構築を許可
    positional_tracking_params.set_floor_as_origin = False  # 原点設定（必要ならTrue）
    positional_tracking_params.enable_imu_fusion = True     # IMUを統合
    err = zed.enable_positional_tracking(positional_tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking:", err)
        zed.close()
        exit(1)

    # 床を原点にする
    positional_tracking_params.set_floor_as_origin = True

    # --- 空間マッピング（Spatial Mapping）の設定 ---
    spatial_mapping_params = sl.SpatialMappingParameters(
        resolution=sl.MAPPING_RESOLUTION.MEDIUM,
        mapping_range=sl.MAPPING_RANGE.MEDIUM,
        max_memory_usage=2048,
        save_texture=False,
        use_chunk_only=True,
        reverse_vertex_order=False,
        map_type=sl.SPATIAL_MAP_TYPE.MESH
    )
    pymesh = sl.Mesh()

    # Grab() で使うランタイムパラメータ
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    # 画像・点群・ポーズを格納するためのオブジェクト生成
    image = sl.Mat()
    point_cloud = sl.Mat()
    pose = sl.Pose()

    # 位置推定の基準（原点）をリセット
    init_params_pose = sl.Transform()
    zed.reset_positional_tracking(init_params_pose)

    # SpatialMapping の詳細パラメータを再設定
    spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    spatial_mapping_params.use_chunk_only = True
    # spatial_mapping_params.save_texture = True
    spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH

    # 空間マッピングを有効化
    returned_state = zed.enable_spatial_mapping(spatial_mapping_params)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Spatial Mapping : " + repr(returned_state) + ". Exit program.")
        exit()

    # OpenGL ビューワ（ogl_viewer）を生成
    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, pymesh, 1)

    # メッシュ／点群データをクリア
    pymesh.clear()
    viewer.clear_current_mesh()

    last_call = time.time()

    Exit = False # SVOの終端に到達したかどうかのフラグ

    pre_timestamp = None # 前フレームのタイムスタンプ（ms）

    speed = opt.speed # SVO再生速度調整用パラメータ

    frame_count = 0  # 処理したフレーム数カウンタ
    frame_step = opt.frame_step # 処理するフレームの間隔

    # メインループ
    while viewer.is_available():

        grab_svo = zed.grab(runtime_params)
        if grab_svo == sl.ERROR_CODE.SUCCESS:

            frame_count += 1
            if frame_count % frame_step != 0:
                continue  # 指定されたフレーム間隔で処理をスキップ

            zed.retrieve_image(image, sl.VIEW.LEFT)

            # # SVO再生速度調整
            # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT) # 現在のフレームのタイムスタンプを取得
            # ms_timestamp = timestamp.get_milliseconds() # タイムスタンプをミリ秒単位で取得
            # if pre_timestamp is not None:
            #     delta = ms_timestamp - pre_timestamp
            #     wait_time = (delta / 1000.0) / speed
            #     if delta > 0:
            #         time.sleep(wait_time)
            # pre_timestamp = ms_timestamp

            tracking_state = zed.get_position(pose)
            mapping_state = zed.get_spatial_mapping_state()
            duration = time.time() - last_call
            if duration > .5 and viewer.chunks_updated():
                zed.request_spatial_map_async()
                last_call = time.time()
            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_spatial_map_async(pymesh)
                viewer.update_chunks()
            change_state = viewer.update_view(image, pose.pose_data(), tracking_state, mapping_state)

        elif grab_svo == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            Exit = True
            print("End of SVO reached. Exiting loop.")
            break

        if keyboard.is_pressed('esc'):
            print("Exiting loop(esc pressed).")
            break

    viewer.clear_current_mesh()   

    # 結果抽出
    zed.extract_whole_spatial_map(pymesh)

    # if Exit:
    # メッシュフィルタリング
    filter_params = sl.MeshFilterParameters()
    filter_params.set(sl.MESH_FILTER.MEDIUM)
    pymesh.filter(filter_params, True)

    if spatial_mapping_params.save_texture:
        print("Save texture set to : {}".format(spatial_mapping_params.save_texture))
        pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

    mesh_dir = opt.mesh_dir
    output_mesh_file = opt.output_mesh_file 
    mesh_path = f"..\..\..\ZED\{mesh_dir}\mesh\{output_mesh_file}"
    
    status = pymesh.save(mesh_path)
    if status:
        print("Mesh saved under " + mesh_path)
    else:
        print("Failed to save the mesh under " + mesh_path)

    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # 解放処理
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    pymesh.clear()
    image.free()
    point_cloud.free()
    zed.close()

def parse_args(init_params):

    svo_dir = opt.svo_dir
    input_svo_file = opt.input_svo_file  
    svo_path = f"..\..\..\ZED\{svo_dir}\svo\{input_svo_file}"
    
    if len(svo_path) > 0 and (svo_path.endswith(".svo") or svo_path.endswith(".svo2")):
        init_params.set_from_svo_file(svo_path)
        print("[Sample] Using SVO File input: {0}".format(svo_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo_dir', default= "samples")
    parser.add_argument('--input_svo_file', default= "svo_sample.svo2")
    parser.add_argument('--mesh_dir', default= "samples")
    parser.add_argument('--output_mesh_file', default= "mesh_sample.obj")
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--frame_step', type=int, default=1)
    opt = parser.parse_args()
    main()

