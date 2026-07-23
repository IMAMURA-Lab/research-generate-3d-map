"""
transforms

役割:
    LiDAR点群(センサー座標系)を、その瞬間のカメラ姿勢(camera_pose)を用いて
    ワールド座標系(mesh・Unity出力で使う共通座標系)へ変換する。
"""


def transform_to_world(lidar_points, camera_pose, calibration=None):
    """
    LiDAR点群をワールド座標系に変換する。

    Args:
        lidar_points: センサー座標系での点群(N x 3)
        camera_pose: その瞬間のZEDカメラの姿勢(ワールド座標系における位置・回転)
        calibration: Calibration インスタンス(LiDAR->ZED間の外部パラメータ)

    Returns:
        ワールド座標系に変換された点群(N x 3)
    """
    raise NotImplementedError
