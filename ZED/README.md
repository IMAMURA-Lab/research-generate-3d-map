# ZED/LiDAR 計測パイプライン

## 構成

```
src/
├─ capture/         現場でセンサーを動かして計測する処理(SVO/LiDAR記録)
├─ preprocessing/    記録済みデータの読み込み・同期・キャリブレーション読込
├─ reconstruction/   点群統合・座標変換・メッシュ生成
├─ detection/        YOLOによる物体検出
├─ localization/     3D位置推定・物体ID管理(MOT)
├─ export/           mesh・objectsの出力(Unity向け)
├─ common/           設定読み込み等の共通処理
└─ pipeline/         上記を順に呼び出す本体処理(rescue_pipeline.py)

scripts/    実行用の薄いエントリポイント
tests/      pytestによる単体・結合テスト
configs/    各処理の設定(yaml)
data/       セッションごとの計測データ・出力(mesh.obj / objects.json)
```

## 実行の流れ(想定)

1. `scripts/run_record.py` — 現場でZED/LiDARを起動し計測(記録)する
2. `scripts/run_sync_check.py` — 記録データの時刻同期状態を確認する
3. `scripts/run_pipeline.py` — 前処理〜メッシュ生成〜物体位置決定〜出力を一括実行する

## 現在の状態

各モジュールはクラス・関数の雛形のみで、処理本体(`NotImplementedError`部分)は未実装。
`docs/研究の開発プラン.md` のチェックリストに沿って、`preprocessing/frame_synchronizer.py` の
同期処理と `localization/object_tracker.py` のMOT実装から着手する想定。

## 未確定事項

- LiDAR機種(選定待ち。決定後 `capture/lidar_recorder.py` ・ `preprocessing/lidar_reader.py` を実装)
- 3D代表座標の求め方(bbox中心 or マスク領域の中央値等)
- Tracker3Dの距離しきい値・更新方式(configs/tracking.yaml で調整予定)
