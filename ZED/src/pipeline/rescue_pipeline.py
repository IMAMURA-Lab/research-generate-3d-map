"""
rescue_pipeline

役割:
    計測パイプライン全体(前処理〜メッシュ生成〜物体位置決定〜出力)を
    順に実行する司令塔。scripts/run_pipeline.py から呼び出される。
"""

from src.preprocessing.svo_reader import SVOReader
from src.preprocessing.lidar_reader import LiDARReader
from src.preprocessing.calibration_loader import Calibration
from src.preprocessing.frame_synchronizer import synchronized_frames
from src.reconstruction.point_cloud_integrator import PointCloudIntegrator
from src.reconstruction.transforms import transform_to_world
from src.detection.yolo_detector import ObjectDetector
from src.localization.object_localizer import ObjectLocalizer
from src.localization.object_tracker import Tracker3D
from src.export.exporter import save_mesh, save_objects


def run(session_dir: str, model_path: str, output_dir: str) -> None:
    """
    Args:
        session_dir: 対象セッションのデータディレクトリ(例: data/session_001)
        model_path: 学習済みYOLOモデルのパス
        output_dir: 出力先ディレクトリ(例: data/map)
    """
    # 1. データ読み込み
    svo_reader = SVOReader(f"{session_dir}/video.svo2")
    lidar_reader = LiDARReader(f"{session_dir}/lidar/")
    calibration = Calibration(f"{session_dir}/calibration.yaml")

    # 2. 各処理モジュールの初期化
    integrator = PointCloudIntegrator()
    detector = ObjectDetector(model_path=model_path)
    localizer = ObjectLocalizer(calibration)
    tracker = Tracker3D()

    # 3. フレームごとに処理
    for frame in synchronized_frames(svo_reader, lidar_reader):
        rgb_image = frame.rgb_image
        lidar_points = frame.lidar_points
        camera_pose = frame.camera_pose

        # 点群をワールド座標へ変換し、メッシュ用に統合
        world_points = transform_to_world(lidar_points, camera_pose, calibration)
        integrator.add_points(world_points)

        # 物体検出
        detections = detector.detect(rgb_image)

        # 各検出結果に3D座標を付与
        measurements = []
        for det in detections:
            object_position = localizer.estimate_position(
                bbox=det.bbox,
                lidar_points=lidar_points,
                camera_pose=camera_pose,
            )
            if object_position is None:
                continue
            measurements.append({
                "class_name": det.class_name,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "position": object_position,
            })

        # 物体IDを更新
        tracker.update(measurements)

    # 4. メッシュ生成
    mesh = integrator.build_mesh()

    # 5. object_idごとの最終位置を決定
    objects = tracker.get_final_objects()

    # 6. 出力
    save_mesh(mesh, f"{output_dir}/mesh.obj")
    save_objects(objects, f"{output_dir}/objects.json")
