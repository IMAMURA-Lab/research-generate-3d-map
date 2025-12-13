# 作成したデータセットの画像とラベルが対応しているか確認するスクリプト
# 実行例
# python check_dataset.py
# --model_name : 確認したいモデル名を指定

import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    model_name = Path(args.class_name)

    images_train_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/images/train"
    images_val_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/images/val"
    labels_train_dir = f"../../../ZED/label_studio_project/work/{model_name}/dataset/labels/train"
    labels_val_dir = f".../../../ZED/label_studio_project/work/{model_name}/dataset/labels/val"

    imgs_train = sorted(os.listdir(images_train_dir))
    lbls_train = sorted(os.listdir(labels_train_dir))

    print(f"train:{[os.path.splitext(f)[0] for f in imgs_train] == [os.path.splitext(f)[0] for f in lbls_train]}")

    imgs_val = sorted(os.listdir(images_val_dir))
    lbls_val = sorted(os.listdir(labels_val_dir))

    print(f"val:{[os.path.splitext(f)[0] for f in imgs_val] == [os.path.splitext(f)[0] for f in lbls_val]}")

if __name__ == "__main__":
    main()
