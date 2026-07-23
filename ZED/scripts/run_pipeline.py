"""
計測パイプライン全体(前処理〜メッシュ生成〜物体位置決定〜出力)を実行する。
"""

from src.pipeline.rescue_pipeline import run


def main():
    run(
        session_dir="data/session_001",
        model_path="models/best.pt",
        output_dir="data/map",
    )


if __name__ == "__main__":
    main()
