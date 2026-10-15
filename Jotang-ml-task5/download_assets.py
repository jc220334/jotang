"""下载任务 5 需要的公开资源：CLIP(RN50) 预训练权重 + 若干张公开数据集图片。

资料来源（公开数据集 / 公开权重）：
  - CLIP RN50 权重：OpenAI CLIP 官方仓库 https://github.com/openai/CLIP
    权重 URL: https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
  - 图片：COCO val2017 数据集（http://cocodataset.org），
    直接用官方图床 http://images.cocodataset.org/val2017/<id>.jpg 下载。

运行方式：python download_assets.py
"""

from __future__ import annotations

import sys
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
IMAGE_DIR = HERE / "image"
RAW_DIR.mkdir(exist_ok=True, parents=True)

RN50_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
)
COCO_BASE = "http://images.cocodataset.org/val2017/{}.jpg"

# 4 张内容互不相同的实验用图（先多下几张候选图、人工看过之后挑出来的这 4 张）
COCO_IDS = [
    "000000039769",  # 粉红色沙发上的两只猫      -> 01_cat.jpg
    "000000000285",  # 草地上的棕熊              -> 02_bear.jpg
    "000000001584",  # 街上的红色双层巴士        -> 03_bus.jpg
    "000000002157",  # 覆盆子蓝莓生日蛋糕        -> 04_cake.jpg
]

# 下载下来的原图 -> clip_pairing.py 实际使用的文件名
RENAME = {
    "000000039769": "01_cat.jpg",
    "000000000285": "02_bear.jpg",
    "000000001584": "03_bus.jpg",
    "000000002157": "04_cake.jpg",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} 已存在（{dest.stat().st_size / 1024:.1f} KB）")
        return
    print(f"[get ] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"[ok  ] -> {dest} ({dest.stat().st_size / 1024:.1f} KB)")


def download_weights() -> None:
    """把 RN50 权重放进 ~/.cache/clip，clip.load('RN50') 就会直接命中缓存。"""
    cache = Path.home() / ".cache" / "clip"
    cache.mkdir(parents=True, exist_ok=True)
    download(RN50_URL, cache / "RN50.pt")


def download_images() -> None:
    for cid in COCO_IDS:
        try:
            download(COCO_BASE.format(cid), RAW_DIR / f"{cid}.jpg")
        except Exception as exc:  # 网络不稳时允许部分失败
            print(f"[fail] {cid}: {exc}", file=sys.stderr)


def arrange_images() -> None:
    """把 4 张原图复制成实验用的固定文件名（已存在就不覆盖）。"""
    IMAGE_DIR.mkdir(exist_ok=True, parents=True)
    for cid, name in RENAME.items():
        src, dst = RAW_DIR / f"{cid}.jpg", IMAGE_DIR / name
        if dst.exists():
            print(f"[skip] {dst.name} 已存在")
        elif src.exists():
            shutil.copy2(src, dst)
            print(f"[copy] {src.name} -> image/{dst.name}   （来源：COCO val2017 图片 ID {cid}）")


if __name__ == "__main__":
    download_weights()
    download_images()
    arrange_images()
