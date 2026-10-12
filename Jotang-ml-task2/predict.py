# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 2 · 实践 2 —— 第 8 题：独立的推理程序

做三件事：
  1. 读入一张本地图片；
  2. 用与训练时**完全一致**的预处理把它变成网络要的 (1,3,128,128) 张量
     （统一 convert('RGB') → 短边缩到 146 → 中心裁 128×128 → /255 → 减均值除方差）；
  3. 输出“猫 / 狗”和概率，并把“预处理后的图片”和预测结果画在一张图里。

用法：
    conda activate jotang-ml
    python predict.py 图片.jpg
    python predict.py 图片1.jpg 图片2.jpg --out 13_inference_demo.png
    python predict.py 图片.jpg --ckpt runs/final.pth --tta

说明：程序只依赖 runs/final.pth（权重文件里已经存了网络结构参数和归一化统计），
     所以它自己就能跑，不需要训练脚本或数据集。
"""

# --------------------------- 环境自举（同训练脚本） ---------------------------
import os as _os
import sys as _sys

_ENV_ROOT = _os.path.dirname(_sys.executable)
for _sub in ("Library\\bin", "Library\\mingw-w64\\bin", "Library\\usr\\bin",
             "DLLs", "Scripts", "bin"):
    _dll_dir = _os.path.join(_ENV_ROOT, _sub)
    if _os.path.isdir(_dll_dir):
        _os.environ["PATH"] = _dll_dir + _os.pathsep + _os.environ.get("PATH", "")
        if hasattr(_os, "add_dll_directory"):
            try:
                _os.add_dll_directory(_dll_dir)
            except OSError:
                pass
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_MPL_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "cache128", "mpl")
_os.makedirs(_MPL_DIR, exist_ok=True)
_os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import argparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
DEFAULT_CKPT = _os.path.join(HERE, "runs", "final.pth")
IMG = 128
CACHE_SIDE = 146               # 与训练时的缓存一致
CLASS_CN = {0: "猫", 1: "狗"}


# --------------------------- 网络结构（必须与训练时完全一致） ---------------------------
def conv_bn(i, o, stride=1):
    return nn.Sequential(
        nn.Conv2d(i, o, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


class CatDogCNN(nn.Module):
    def __init__(self, width=32, dropout=0.0, nclass=2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn(3, w, stride=2),
            conv_bn(w, w),
            nn.MaxPool2d(2),
            conv_bn(w, w * 2),
            conv_bn(w * 2, w * 2),
            nn.MaxPool2d(2),
            conv_bn(w * 2, w * 4),
            conv_bn(w * 4, w * 4),
            nn.MaxPool2d(2),
            conv_bn(w * 4, w * 8),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 8 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nclass),
        )

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------- 预处理（与训练时一致） ---------------------------
def preprocess(im, mean, std):
    """一张 PIL 图片 → (1,3,128,128) 的归一化张量 + 供肉眼检查的 128×128 图片。"""
    im = im.convert("RGB")                                    # 灰度/P/CMYK/RGBA 统统先转 RGB
    w, h = im.size
    s = CACHE_SIDE / min(w, h)
    if s < 1:                                                 # 只缩小不放大（与训练脚本一致）
        im = im.resize((max(CACHE_SIDE, int(round(w * s))),
                        max(CACHE_SIDE, int(round(h * s)))), Image.BILINEAR)
    w2, h2 = im.size
    left, top = (w2 - IMG) // 2, (h2 - IMG) // 2
    im = im.crop((left, top, left + IMG, top + IMG))
    view = np.asarray(im, dtype=np.uint8)                     # 给人看的“处理后的图片”
    x = torch.from_numpy(np.array(view, copy=True)).permute(2, 0, 1).float().div_(255.0)
    x = (x - torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)) \
        / torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return x.unsqueeze(0), view


@torch.no_grad()
def infer(model, im, mean, std, tta=False):
    x, view = preprocess(im, mean, std)
    p = torch.softmax(model(x), 1)[0]
    if tta:                                                   # 水平翻转各算一次再平均
        p = (p + torch.softmax(model(torch.flip(x, dims=[3])), 1)[0]) / 2
    return p.numpy(), view


def main():
    ap = argparse.ArgumentParser(description="猫狗分类推理程序")
    ap.add_argument("images", nargs="+", help="一张或多张本地图片")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="把结果画成图片保存（默认 predict_result.png）")
    ap.add_argument("--tta", action="store_true", help="水平翻转 TTA，两次预测取平均")
    args = ap.parse_args()

    if not _os.path.exists(args.ckpt):
        raise SystemExit(f"找不到权重文件 {args.ckpt}，请先运行 catdog_classifier.py --stages train --tag final")
    torch.set_num_threads(4)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = CatDogCNN(width=ckpt["arch"]["width"], dropout=ckpt["arch"]["dropout"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    mean, std = ckpt["norm"]["mean"], ckpt["norm"]["std"]
    print(f"权重：{args.ckpt}（width={ckpt['arch']['width']}，"
          f"验证集最好准确率 {max(e['va_acc'] for e in ckpt['history']['epochs']):.4f}）")

    n = len(args.images)
    fig = plt.figure(figsize=(4.6 * n, 7.6)) if n > 1 else plt.figure(figsize=(12, 5.2))
    results = []
    for k, path in enumerate(args.images):
        im = Image.open(path)
        p, view = infer(model, im, mean, std, tta=args.tta)
        lab = int(p.argmax())
        w, h = im.size
        results.append((path, lab, p))
        print(f"\n图片：{path}（原始 {w}×{h}，{im.mode}）")
        print(f"  判断：{CLASS_CN[lab]}" + ("（Dog）" if lab else "（Cat）"))
        print(f"  概率：猫 {p[0]:.4f}，狗 {p[1]:.4f}"
              + (f"，TTA 置信度差距 {abs(p[0] - p[1]):.4f}" if args.tta else ""))
        print(f"  预处理：convert('RGB') → 短边缩到 {CACHE_SIDE} → 中心裁 {IMG}×{IMG} → "
              f"/255 → (x-mean)/std，张量形状 {tuple(preprocess(im, mean, std)[0].shape)}")

        if n == 1:
            ax0 = fig.add_subplot(1, 3, 1)
            ax1 = fig.add_subplot(1, 3, 2)
            ax2 = fig.add_subplot(1, 3, 3)
        else:
            ax0 = fig.add_subplot(n, 3, 3 * k + 1)
            ax1 = fig.add_subplot(n, 3, 3 * k + 2)
            ax2 = fig.add_subplot(n, 3, 3 * k + 3)
        ax0.imshow(im.convert("RGB"))
        ax0.axis("off")
        ax0.set_title(f"输入图片\n{w}×{h}（{im.mode}）", fontsize=10)
        ax1.imshow(view)
        ax1.axis("off")
        ax1.set_title("程序内预处理后的 128×128\n（给网络的就是这张）", fontsize=10)
        if n == 1:
            ax2.axis("off")
            ax2.text(0.5, 0.78, f"预测：{CLASS_CN[lab]}", ha="center", fontsize=22,
                     color="#C44E52" if lab else "#4C72B0")
            txt = "\n".join([f"猫 Cat：{p[0]:.4f}", f"狗 Dog：{p[1]:.4f}"])
            ax2.text(0.5, 0.5, txt, ha="center", fontsize=14)
        else:
            ax2.barh([0, 1], [p[0], p[1]], color=["#4C72B0", "#C44E52"])
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(["猫", "狗"])
            ax2.set_xlim(0, 1)
            ax2.set_title(f"预测：{CLASS_CN[lab]}\n猫 {p[0]:.3f} / 狗 {p[1]:.3f}", fontsize=10)
    fig.suptitle("任务 8：独立推理程序 predict.py 的输出"
                 + ("（--tta 开启水平翻转 TTA）" if args.tta else ""), fontsize=14)
    fig.tight_layout()
    out = args.out or _os.path.join(HERE, "predict_result.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n结果图已保存：{out}")


if __name__ == "__main__":
    main()
