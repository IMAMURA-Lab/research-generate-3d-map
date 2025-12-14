# 作成したデータセットの画像とラベルが対応しているか確認するスクリプト
# 実行例
# python check_dataset.py
# --model_name: 学習モデル名（label_studio_project/work/以下のフォルダ名）

import os
import argparse
from pathlib import Path

def main():

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    opt = parser.parse_args()

    model_name = opt.model_name # 学習モデル名

    # -----------------------------
    # 使用するデータのパス指定
    # -----------------------------
    images_train_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/images/train"
    images_val_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/images/val"
    labels_train_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/labels/train"
    labels_val_dir = f".../../../ZED/label_studio_project/work/{model_name}/dataset/labels/val"

    # 学習用データの互換性チェック
    imgs_train = sorted(os.listdir(images_train_dir))
    lbls_train = sorted(os.listdir(labels_train_dir))
    print(f"train:{[os.path.splitext(f)[0] for f in imgs_train] == [os.path.splitext(f)[0] for f in lbls_train]}")

    # 検証用データの互換性チェック
    imgs_val = sorted(os.listdir(images_val_dir))
    lbls_val = sorted(os.listdir(labels_val_dir))
    print(f"val:{[os.path.splitext(f)[0] for f in imgs_val] == [os.path.splitext(f)[0] for f in lbls_val]}")

if __name__ == "__main__":

    main()
