# HEIC形式の画像をPNG形式に変換するスクリプト
# 実行例
# python convert_heic_to_png.py
# --class_name: クラス名（label_studio_project/data/classes/以下のフォルダ名）

from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
import argparse
from datetime import datetime

register_heif_opener() # HEICを開けるように設定

def convert_heic_to_png(src_folder, dest_folder):

    src = Path(src_folder)
    dest = Path(dest_folder)

    for heic_file in src.glob("*.heic"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = dest / (f"image_{ts}.png")

        with Image.open(heic_file) as img:
            img.save(png_path, "PNG")
            print(f"変換完了: {png_path}")

def main():

    # -----------------------------
    # 引数処理
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", required=True)
    opt = parser.parse_args()
    class_name = opt.class_name # クラス名

    # -----------------------------
    # 変換元と変換先のパス指定
    # -----------------------------
    source = f"../../../ZED/label_studio_project/data/classes/{class_name}/images/new/heic"
    output = f"../../../ZED/label_studio_project/data/classes/{class_name}/images/new"

    convert_heic_to_png(source, output)

if __name__ == "__main__":

    main()

