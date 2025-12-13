# データセットを学習用と検証用に分割するスクリプト
# 実行例
# python divide_dataset.py
# --model_name : 分割対象のモデル名（データセットの元フォルダ名）
# --val-ratio : 評価用データの割合（デフォルト 0.2）
# --seed : 乱数シード（デフォルト 42）
# --mode : copy または move（デフォルト move）

import argparse
import os
import random
import shutil
from pathlib import Path

def collect_pairs(images_dir, labels_dir, exts=(".jpg",".png",".jpeg")):
    # images_dir 内の画像ファイルを拡張子リスト exts で走査し、
    # 各画像に対して同名の .txt ラベルファイルが labels_dir に存在するかを確認します。
    # 存在する場合は (画像パス, ラベルパス) のタプルを pairs に追加します。
    # 存在しない場合は警告を出力してその画像をスキップします。
    # 最後に pairs を名前順にソートして返します。

    images = []
    for ext in exts:
        # Path.glob はワイルドカードでファイルを列挙します。例えば images_dir.glob("*.jpg").
        images.extend(Path(images_dir).glob(f"*{ext}"))
    pairs = []
    for img in images:
        # img.stem は拡張子を除いたファイル名（例: '0001.jpg' -> '0001'）を返します。
        label = Path(labels_dir) / (img.stem + ".txt")
        if label.exists():
            # 画像と対応ラベルが両方あるペアだけを扱います。
            pairs.append((str(img), str(label)))
        else:
            # ラベルファイルが無い画像はデータ品質の問題となるので警告を出す。
            print(f"WARNING: label missing for image {img.name}, skipping.")
    # ソートしておくと分割が毎回同じ順序で行われやすく、seed によるシャッフルの挙動が安定します。
    pairs.sort()
    return pairs

def split_pairs(pairs, val_ratio, seed):
    # pairs をランダムにシャッフルして train と val に分割します。
    # seed を受け取り reproducible（再現可能）なシャッフルを実現します。
    # val_ratio は検証用データの割合（0.0 - 1.0）です。

    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    # n_val を計算します。データが1枚しかない場合は検証用を0にする（n>1 のときのみ計算）
    # また、少数のデータでも最低1件は val に入れたい場合は max(1, ...) のままにできます。
    # ここでは n>1 の場合に max(1, int(round(n * val_ratio))) を採用しているため、
    # 2枚以上のときは少なくとも1枚は val に入る可能性があります。用途に応じて調整してください。
    n_val = max(1, int(round(n * val_ratio))) if n>1 else 0
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val

def copy_or_move(pairs, dst, mode="move"):
    for img_path, lbl_path in pairs:
        dst_img = Path(dst) / ("images/train" if "train" in dst_img_temp == img_path else "images")  # placeholder

def perform(pairs, dst, set_name, mode="move"):
    # 実際にファイルをコピーまたは移動する関数
    # pairs: (画像パス, ラベルパス) のリスト
    # dst: 出力先のベースディレクトリ
    # set_name: 'train' または 'val' を想定
    # mode: 'copy'（デフォルト）または 'move'。

    for img_path, lbl_path in pairs:
        # それぞれ images/<set_name>/ と labels/<set_name>/ にファイルを配置する。
        dst_img = Path(dst) / f"images/{set_name}" / Path(img_path).name
        dst_lbl = Path(dst) / f"labels/{set_name}" / Path(lbl_path).name
        if mode == "move":
            # move は元ファイルを削除して移動先にファイルを移動するため、
            # 元のデータを残したくないときに使用します。ファイルシステム上での移動となります。
            shutil.move(img_path, str(dst_img))
            shutil.move(lbl_path, str(dst_lbl))
        else:
            # copy2 はメタデータ（更新時刻など）も可能な限り保持してコピーします。
            shutil.copy2(img_path, str(dst_img))
            shutil.copy2(lbl_path, str(dst_lbl))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("copy","move"), default="move")
    args = parser.parse_args()

    model_name = Path(args.model_name)

    images_dir = f"..\..\..\ZED\label_studio_project/work/{model_name}/annotations/images" # データセット作成に使用する元画像フォルダ
    labels_dir = f"..\..\..\ZED\label_studio_project/work/{model_name}/annotations/labels" # データセット作成に使用する元ラベルフォルダ
    dataset_dir = f"..\..\..\ZED\label_studio_project/work/{model_name}/dataset" # 分割後のデータセット出力先フォルダ

    pairs = collect_pairs(images_dir, labels_dir)
    if len(pairs) == 0:
        print("No valid image-label pairs found.")
        return

    # 分割処理
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    print(f"Total pairs: {len(pairs)}, Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # train, val をそれぞれコピー/移動
    perform(train_pairs, dataset_dir, "train", mode=args.mode)
    perform(val_pairs, dataset_dir, "val", mode=args.mode)

    # print("Done. Counts:")
    # for p in ["images/train","images/val","labels/train","labels/val"]:
    #     # 各ディレクトリ内のファイル数を表示
    #     print(p, len(list((dataset_dir / p).glob("*"))))

if __name__ == "__main__":
    main()
