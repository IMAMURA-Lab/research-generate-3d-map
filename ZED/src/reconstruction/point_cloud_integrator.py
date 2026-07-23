"""
PointCloudIntegrator

役割:
    フレームごとに得られるワールド座標の点群を蓄積し、
    全フレーム処理後に建物構造のメッシュ(mesh.obj相当)を生成する。
"""


class PointCloudIntegrator:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._accumulated_points = []  # TODO: Open3D点群オブジェクト等に置き換え

    def add_points(self, world_points) -> None:
        """1フレーム分のワールド座標点群を統合対象に追加する。"""
        raise NotImplementedError

    def build_mesh(self):
        """蓄積した点群からメッシュを生成する(例: Open3DのTSDF統合/Poisson再構成)。"""
        raise NotImplementedError
