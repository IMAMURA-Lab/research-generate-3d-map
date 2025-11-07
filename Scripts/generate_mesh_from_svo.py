# SVOファイルまたはSVO2ファイルからメッシュを生成するスクリプト

import sys
import time
import pyzed.sl as sl            # pyzed.sl: Stereolabs ZED SDK の Python バインディング。カメラ操作、深度、ポーズ、空間マッピング等を提供。
import ogl_viewer.viewer as gl   # ogl_viewer: OpenGLベースの簡易ビューワー（サンプル付属）。画面表示・ユーザー入力を処理する。
import argparse                  # コマンドライン引数をパースする標準モジュール
import cv2
import keyboard                  # 外部ライブラリ。キーボード入力の状態をポーリングするのに便利（※管理者権限が必要な場合あり）


def main():
    # --- 初期化パラメータの設定 ---
    init_params = sl.InitParameters()

    # コマンドライン引数に基づいて init_params を変更する（SVOファイル入力やストリームなどに対応）
    parse_args(init_params)

    # SVO 再生の実時間モード（True ならSVOのタイムスタンプに従って再生、False ならできるだけ早く再生）
    init_params.svo_real_time_mode = True

    # 深度モードを PERFORMANCE に変更（sl.DEPTH_MODE.PERFORMANCE: 高速重視の深度推定）
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    # 単位（メートル）を指定。これにより、取得される深度や座標系のスケールが決まる。
    init_params.coordinate_units = sl.UNIT.METER

    # 座標系の選択。OpenGL の慣例に合わせて、Y軸が上（Y_UP）の右手系を使う。
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # OpenGL's coordinate system is right_handed

    # 深度（距離）を何メートルまで有効とするか（max distance）。これ以上の物体は深度推定しない。
    init_params.depth_maximum_distance = 8.0

    # カメラオブジェクトを作成して開く
    zed = sl.Camera()
    status = zed.open(init_params)

    # カメラが開けない場合はエラーメッセージを出して終了
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(status) + ". Exit program.")
        exit()

    # トラッキング／マッピングの状態変数（初期はオフ）
    # tracking_state = sl.POSITIONAL_TRACKING_STATE.OFF
    mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # カメラの情報（解像度、焦点距離、センサーサイズなど）が取得可能
    camera_infos = zed.get_camera_information()

    # Positional Tracking の有効化
    positional_tracking_params = sl.PositionalTrackingParameters()
    positional_tracking_params.enable_area_memory = True  # カメラ移動に応じたマップ構築を許可
    positional_tracking_params.set_floor_as_origin = False  # 原点設定（必要ならTrue）
    positional_tracking_params.enable_imu_fusion = True     # IMU統合（ZED Mini / 2 / 2i / 2i+ / Xシリーズでは重要）
    err = zed.enable_positional_tracking(positional_tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to enable positional tracking:", err)
        zed.close()
        exit(1)

    # 床を原点にする
    positional_tracking_params.set_floor_as_origin = True

    # --- 空間マッピング（Spatial Mapping）のための設定 ---
    if opt.build_mesh:
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
    else:
        spatial_mapping_params = sl.SpatialMappingParameters(
            resolution=sl.MAPPING_RESOLUTION.MEDIUM,
            mapping_range=sl.MAPPING_RANGE.MEDIUM,
            max_memory_usage=2048,
            save_texture=False,
            use_chunk_only=True,
            reverse_vertex_order=False,
            map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        )
        pymesh = sl.FusedPointCloud()

    # Grab() で使うランタイムパラメータ
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50

    # 画像・点群・ポーズを格納するための Mat / Pose オブジェクトを生成
    image = sl.Mat()
    point_cloud = sl.Mat()
    pose = sl.Pose()

    # 位置推定の基準（原点）をリセット
    init_params_pose = sl.Transform()
    zed.reset_positional_tracking(init_params_pose)

    # SpatialMapping の詳細パラメータを再設定
    spatial_mapping_params.resolution_meter = sl.SpatialMappingParameters().get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
    spatial_mapping_params.use_chunk_only = True
    spatial_mapping_params.save_texture = True
    if opt.build_mesh:
        spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
    else:
        spatial_mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD

    # 空間マッピングを有効化
    returned_state = zed.enable_spatial_mapping(spatial_mapping_params)
    if returned_state != sl.ERROR_CODE.SUCCESS:
        print("Enable Spatial Mapping : " + repr(returned_state) + ". Exit program.")
        exit()

    # OpenGL ビューワ（ogl_viewer）を生成
    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, pymesh, int(opt.build_mesh))

    # メッシュ／点群データをクリア
    pymesh.clear()
    viewer.clear_current_mesh()

    last_call = time.time()

    # メインループ
    while viewer.is_available():
        grab_svo = zed.grab(runtime_params)
        if grab_svo == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
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
            print("End of SVO reached. Exiting loop.")
            break

        if keyboard.is_pressed('esc'):
            print("Stopping recording after 1000 frames.")
            break

    # マッピング停止 → 結果抽出
    zed.extract_whole_spatial_map(pymesh)

    if opt.build_mesh:
        filter_params = sl.MeshFilterParameters()
        filter_params.set(sl.MESH_FILTER.MEDIUM)
        pymesh.filter(filter_params, True)
        viewer.clear_current_mesh()

        if spatial_mapping_params.save_texture:
            print("Save texture set to : {}".format(spatial_mapping_params.save_texture))
            pymesh.apply_texture(sl.MESH_TEXTURE_FORMAT.RGBA)

        filepath = "..\Materials\mesh_sample.obj"
        status = pymesh.save(filepath)
        if status:
            print("Mesh saved under " + filepath)
        else:
            print("Failed to save the mesh under " + filepath)

        mapping_state = sl.SPATIAL_MAPPING_STATE.NOT_ENABLED

    # 解放処理
    image.free(memory_type=sl.MEM.CPU)
    pymesh.clear()
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    pymesh.clear()
    image.free()
    point_cloud.free()
    zed.close()


def parse_args(init_params):
    """ コマンドライン引数に基づいて init（InitParameters）を書き換える関数 """
    if len(opt.input_svo_file) > 0 and (opt.input_svo_file.endswith(".svo") or opt.input_svo_file.endswith(".svo2")):
        init_params.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it', default='')
    parser.add_argument('--build_mesh', help='Either the script should plot a mesh or point clouds of surroundings', action='store_true')
    opt = parser.parse_args()
    main()
