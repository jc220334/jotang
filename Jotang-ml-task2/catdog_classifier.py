# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 2 · 实践 2：训练猫狗分类器

对应课件里的 9 个问题（本脚本负责 1~7、9 题，第 8 题的推理程序是独立的 predict.py）：
  任务 1：观察数据集 —— 猫狗各多少张、随机展示样本、尺寸/清晰度/背景是否一致
  任务 2：数据读取与预处理流程 —— 怎样组成一个 batch、预处理前后对比、检查拉伸/裁剪/颜色异常
  任务 3：搭建一个包含卷积层、激活函数和池化层的 CNN
  任务 4：记录一张图片经过 CNN 各层后的 Tensor 形状，说明宽高通道数的变化
  任务 5：把第一层 / 最后一层卷积输出的特征图可视化，比较不同通道关注什么
  任务 6：数据增强的对照实验（无增强 / 水平翻转 / 随机裁剪 / 颜色扰动 / 全部）
  任务 7：查看预测错误的样本、分析原因并改进
  任务 9：绘制模型结构图

数据：data/PetImages/{Cat,Dog}/*.jpg（Kaggle cats vs dogs，共 25000 张）

运行方式（分阶段，方便一步步看结果）：
    conda activate jotang-ml
    python catdog_classifier.py --stages cache           # 建缓存（只在第一次需要）
    python catdog_classifier.py --stages explore         # 任务 1
    python catdog_classifier.py --stages preprocess      # 任务 2
    python catdog_classifier.py --stages arch            # 任务 3、4、9
    python catdog_classifier.py --stages features        # 任务 5
    python catdog_classifier.py --stages augment         # 任务 6（增强样例图）
    python catdog_classifier.py --stages train --tag noaug --aug "" --epochs 8
    python catdog_classifier.py --stages train --tag aug --aug flip,crop,color --epochs 8
    python catdog_classifier.py --stages report          # 汇总曲线图
    python catdog_classifier.py --stages errors          # 任务 7
    python catdog_classifier.py --stages all             # 一把梭（时间较长）

输出图片（默认放在本脚本同目录）：
    05_dataset_overview.png  数据集观察
    06_preprocess.png        预处理流程与前后对比
    07_shapes.png            各层 Tensor 形状
    08_feature_maps.png      卷积特征图可视化
    09_augmentation.png      数据增强样例
    10_training_curves.png   训练曲线与对照实验汇总
    11_error_analysis.png    错误样本分析
    12_model_arch.png        模型结构图

中间产物：
    data/cache128/images_128.npy   预处理后的图片缓存（(N,128,128,3) uint8，内存映射读取）
    data/cache128/meta.csv         每张图的原始尺寸、色彩模式、是否损坏
    data/cache128/splits.npz       训练/验证/测试划分
    data/cache128/norm.json        训练集均值方差
    runs/<tag>.json / <tag>.pth    每次训练的记录与权重
"""

# ==================================================================
# 【第 0 步】运行环境自举（与 conv_kernel.py 完全一致的坑，不要删）
# ------------------------------------------------------------------
# 1) 不激活 conda 环境直接跑 python.exe 时，numpy 的 MKL 找不到 DLL，
#    会原生崩溃报 "Windows fatal exception: code 0xc06d007f"。
# 2) 本环境同时存在两份 OpenMP（MKL 的 libomp.dll 与 torch 的 libiomp5md.dll），
#    torch 先加载之后 numpy 一调 BLAS 就报 "OMP: Error #15" 并中止进程。
# ==================================================================
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
# 3) 当前进程可能没有写用户目录（%APPDATA%）的权限，matplotlib 存字体缓存时会报
#    "Could not save font_manager cache ... Permission denied"。把缓存目录指到项目里，
#    既消掉警告，也能保证中文字体能被正常找到（否则中文会画成方框）。
_MPL_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "cache128", "mpl")
_os.makedirs(_MPL_DIR, exist_ok=True)
_os.environ.setdefault("MPLCONFIGDIR", _MPL_DIR)

import argparse
import csv
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(HERE, "data")
PET_DIR = _os.path.join(DATA_DIR, "PetImages")
CACHE_DIR = _os.path.join(DATA_DIR, "cache128")
RUN_DIR = _os.path.join(HERE, "runs")

CLASSES = ("Cat", "Dog")
CLASS_CN = {"Cat": "猫", "Dog": "狗"}
IMG = 128                     # 网络输入边长
CACHE_SIDE = 146              # 缓存时先把短边缩到 146，再中心裁到 128（1.14 倍，和 torchvision 256→224 一致）
SEED = 42
LINE = "=" * 74
MEAN_PATH = _os.path.join(CACHE_DIR, "norm.json")


def section(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================================================
# 一、数据缓存：把 25000 张大小不一的 JPEG 统一成 (N,128,128,3) uint8
# ------------------------------------------------------------------
# 为什么不训练时直接读 JPEG：每张解码 + 缩放约 6~8 ms，一个 epoch 光读图就要
# 两三分钟，而且会和训练线程抢 CPU。只解析一次、存成连续数组后，取一张图就是
# 一次内存切片，训练才跑得动。
# ==================================================================
def _load_one(path):
    """读一张图，返回 (缩放后的 uint8 数组, 原始宽, 原始高, 色彩模式)。"""
    im = Image.open(path)
    im.load()
    mode, w, h = im.mode, im.size[0], im.size[1]
    im = im.convert("RGB")                                    # 统一转 RGB：灰度/P/CMYK/RGBA 都能转
    s = CACHE_SIDE / min(w, h)
    if s < 1:                                                 # 只缩小不放大，避免把小图放大成模糊块
        im = im.resize((max(CACHE_SIDE, int(round(w * s))),
                        max(CACHE_SIDE, int(round(h * s)))), Image.BILINEAR)
    w2, h2 = im.size
    left = (w2 - IMG) // 2                                    # 中心裁剪：只切掉多余边缘，不拉伸
    top = (h2 - IMG) // 2
    im = im.crop((left, top, left + IMG, top + IMG))
    return np.asarray(im, dtype=np.uint8), w, h, mode


def build_cache(force=False):
    section("建立缓存：data/cache128/images_128.npy")
    _os.makedirs(CACHE_DIR, exist_ok=True)
    arr_path = _os.path.join(CACHE_DIR, "images_128.npy")
    meta_path = _os.path.join(CACHE_DIR, "meta.csv")
    if _os.path.exists(arr_path) and _os.path.exists(meta_path) and not force:
        with open(meta_path, encoding="utf-8") as f:
            n_ok = sum(1 for row in csv.DictReader(f) if row["ok"] == "1")
        print(f"  已存在缓存，跳过（可用 {n_ok} 张）")
        return

    jobs = []
    for ci, cls in enumerate(CLASSES):
        for i in range(12500):
            jobs.append((ci, cls, i, _os.path.join(PET_DIR, cls, f"{i}.jpg")))
    n_all = len(jobs)
    print(f"  共 {n_all} 个文件，开始解码 / 缩放 / 中心裁剪 ……")

    big = np.lib.format.open_memmap(arr_path, mode="w+",
                                    dtype=np.uint8, shape=(n_all, IMG, IMG, 3))
    rows = [None] * n_all
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:             # PIL 解码在 C 层放掉 GIL，多线程能加速
        fut = {ex.submit(_load_one, j[3]): k for k, j in enumerate(jobs)}
        for f in as_completed(fut):
            k = fut[f]
            ci, cls, i, path = jobs[k]
            try:
                a, w, h, mode = f.result()
                big[k] = a
                rows[k] = dict(row=k, cls=cls, cls_id=ci, name=f"{i}.jpg",
                               orig_w=w, orig_h=h, mode=mode, ok=1, err="")
            except Exception as e:                            # 损坏文件：Cat/666.jpg、Dog/11702.jpg
                rows[k] = dict(row=k, cls=cls, cls_id=ci, name=f"{i}.jpg",
                               orig_w=0, orig_h=0, mode="", ok=0, err=type(e).__name__)
            done += 1
            if done % 5000 == 0:
                print(f"    {done}/{n_all}  ({time.time() - t0:.0f}s)")
    big.flush()
    del big
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "cls", "cls_id", "name",
                                          "orig_w", "orig_h", "mode", "ok", "err"])
        w.writeheader()
        w.writerows(rows)
    n_bad = sum(1 for r in rows if r["ok"] == 0)
    print(f"  完成：{n_all} 个文件，{n_all - n_bad} 张可用，{n_bad} 张损坏，"
          f"耗时 {time.time() - t0:.0f}s")
    print(f"  缓存 {_os.path.getsize(arr_path) / 2**30:.2f} GB（内存映射读取，不占内存）")


def load_meta():
    with open(_os.path.join(CACHE_DIR, "meta.csv"), encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def load_arr():
    return np.load(_os.path.join(CACHE_DIR, "images_128.npy"), mmap_mode="r")


def build_splits():
    """按类分层划分 训练/验证/测试 = 70% / 15% / 15%。"""
    path = _os.path.join(CACHE_DIR, "splits.npz")
    if _os.path.exists(path):
        return dict(np.load(path))
    meta = load_meta()
    rng = np.random.RandomState(SEED)
    tr, va, te = [], [], []
    for ci in range(len(CLASSES)):
        rows = np.array([int(r["row"]) for r in meta
                         if int(r["cls_id"]) == ci and r["ok"] == "1"])
        rng.shuffle(rows)
        n = len(rows)
        n_tr, n_va = int(n * 0.70), int(n * 0.15)
        tr.append(rows[:n_tr])
        va.append(rows[n_tr:n_tr + n_va])
        te.append(rows[n_tr + n_va:])
    out = {"train": np.sort(np.concatenate(tr)), "val": np.sort(np.concatenate(va)),
           "test": np.sort(np.concatenate(te))}
    np.savez(path, **out)
    return out


def get_labels(idx):
    meta = load_meta()
    return np.array([int(meta[int(i)]["cls_id"]) for i in idx])


def compute_norm():
    """统计训练集的逐通道均值/标准差。"""
    if _os.path.exists(MEAN_PATH):
        with open(MEAN_PATH, encoding="utf-8") as f:
            return json.load(f)
    arr = load_arr()
    sp = build_splits()
    idx = sp["train"][::20]                                   # 抽样统计就够
    x = np.asarray(arr[idx], dtype=np.float32) / 255.0
    out = {"mean": x.mean(axis=(0, 1, 2)).tolist(),
           "std": x.std(axis=(0, 1, 2)).tolist()}
    with open(MEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ==================================================================
# 任务 1：观察数据集
# ==================================================================
def stage_explore():
    section("任务 1：观察数据集")
    meta = load_meta()
    arr = load_arr()
    for ci, cls in enumerate(CLASSES):
        allc = [r for r in meta if int(r["cls_id"]) == ci]
        ok = [r for r in allc if r["ok"] == "1"]
        print(f"  {cls:3s}（{CLASS_CN[cls]}）：文件 {len(allc)} 个，可用 {len(ok)} 张")
    bad = [r for r in meta if r["ok"] == "0"]
    print(f"  损坏文件 {len(bad)} 个：" +
          "，".join(f"{r['cls']}/{r['name']}（{r['err']}）" for r in bad))

    ok_rows = [r for r in meta if r["ok"] == "1"]
    w = np.array([int(r["orig_w"]) for r in ok_rows])
    h = np.array([int(r["orig_h"]) for r in ok_rows])
    ar = w / h
    px = w * h
    print(f"  原始尺寸：宽 {w.min()}~{w.max()}（中位数 {int(np.median(w))}），"
          f"高 {h.min()}~{h.max()}（中位数 {int(np.median(h))}）")
    print(f"  像素总数：最小 {int(px.min())}，最大 {int(px.max())}，相差 {px.max() / px.min():.0f} 倍")
    print(f"  长宽比 {ar.min():.2f} ~ {ar.max():.2f}；比例 >1.5 或 <0.67 的极端图有 "
          f"{int(((ar > 1.5) | (ar < 0.67)).sum())} 张")
    modes = {}
    for r in ok_rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    print("  色彩模式：" + "，".join(f"{k} {v} 张" for k, v in sorted(modes.items()))
          + f"　→ 非 RGB 共 {sum(v for k, v in modes.items() if k != 'RGB')} 张，"
            "不显式 convert('RGB') 就会颜色异常")

    # 清晰度：灰度图上的拉普拉斯方差（越小越糊），抽样估计
    ok_idx = np.array([int(r["row"]) for r in ok_rows])
    sample = ok_idx[:: max(1, len(ok_idx) // 4000)][:4000]
    sharp = np.zeros(len(sample), dtype=np.float64)
    for i, row in enumerate(sample):
        g = np.asarray(arr[row], dtype=np.float32).mean(2)
        lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
               - g[1:-1, :-2] - g[1:-1, 2:])
        sharp[i] = lap.var()
    q10, q50, q90 = np.percentile(sharp, [10, 50, 90])
    print(f"  清晰度（拉普拉斯方差，抽样 {len(sample)} 张，在 128×128 上算）：")
    print(f"    10% 分位 {q10:.0f}，中位数 {q50:.0f}，90% 分位 {q90:.0f}，"
          f"最大 {sharp.max():.0f}")
    print(f"    最糊的 10% 图片方差只有 {q10:.0f}（不到中位数的 {q10 / q50:.2f} 倍），"
          "这批图肉眼看起来也明显更糊")
    print("  背景：第一行随机样本可见 —— 室内地板、沙发、草地、雪地、纯色棚拍都有，"
          "没有任何统一性（下面会看到它带来的后果）")

    fig = plt.figure(figsize=(16, 11.6))
    gs = fig.add_gridspec(4, 6, hspace=0.78, wspace=0.3)
    rng = np.random.RandomState(0)
    ok_idx = np.array([int(r["row"]) for r in ok_rows])
    pick = rng.choice(ok_idx, 12, replace=False)
    for k, r in enumerate(pick):
        ax = fig.add_subplot(gs[k // 6, k % 6])
        ax.imshow(arr[r])
        ax.axis("off")
        m = meta[int(r)]
        ax.set_title(f"{CLASS_CN[m['cls']]}  {m['orig_w']}×{m['orig_h']}  {m['mode']}", fontsize=8)

    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(w, bins=60, color="#4C72B0", log=True)
    ax.set_title(f"原始宽度分布（中位数 {int(np.median(w))} px）", fontsize=10)
    ax.set_xlabel("宽 (px)")
    ax.set_ylabel("图片数（对数）")

    ax = fig.add_subplot(gs[2, 2:4])
    ax.hist(np.log10(px), bins=60, color="#55A868")
    ax.set_title("像素总数分布（log10）", fontsize=10)
    ax.set_xlabel("log10(宽×高)")

    ax = fig.add_subplot(gs[2, 4:6])
    hi = np.percentile(sharp, 99)
    ax.hist(np.clip(sharp, 0, hi), bins=60, color="#C44E52")
    ax.set_title(f"清晰度（拉普拉斯方差，中位数 {q50:.0f}）\n"
                 f"直方图截到 99% 分位 {hi:.0f}；左边最糊的 10% 阈值 {q10:.0f}", fontsize=10)
    ax.set_xlabel("方差（越小越糊）")

    ax = fig.add_subplot(gs[3, 0:3])
    ax.bar([1, 2], [12500, 12500], color=["#DD8452", "#4C72B0"], width=0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Cat 猫", "Dog 狗"])
    ax.set_title("两类各 12500 个文件，可用 12499 张\n"
                 "（各损坏 1 张：Cat/666.jpg、Dog/11702.jpg）", fontsize=10)
    ax.set_ylabel("图片数")
    ax.set_ylim(0, 14000)

    ax = fig.add_subplot(gs[3, 3:6])
    ms = sorted(modes.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in ms], [v for _, v in ms], color="#8172B3")
    ax.set_xscale("log")
    ax.set_title(f"色彩模式：非 RGB 共 {sum(v for k, v in modes.items() if k != 'RGB')} 张\n"
                 "（P 调色板 / L 灰度 / CMYK / RGBA）", fontsize=10)
    fig.suptitle("任务 1：数据集观察 —— 尺寸、清晰度、背景都不统一", fontsize=15)
    fig.savefig(_os.path.join(HERE, "05_dataset_overview.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 05_dataset_overview.png")


# ==================================================================
# 任务 2：数据读取与预处理
# ==================================================================
def random_resized_crop(img, out=IMG, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
    """随机裁一块再缩放到 out×out（和 torchvision.RandomResizedCrop 同思路）。"""
    H, W = img.shape[:2]
    area = H * W
    for _ in range(10):
        ta = area * random.uniform(*scale)
        a = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(ta * a))), int(round(math.sqrt(ta / a)))
        if 0 < cw <= W and 0 < ch <= H:
            x0 = random.randint(0, W - cw)
            y0 = random.randint(0, H - ch)
            patch = img[y0:y0 + ch, x0:x0 + cw]
            return np.asarray(Image.fromarray(patch).resize((out, out), Image.BILINEAR))
    return img


def color_jitter(t, b=0.3, c=0.3, s=0.3):
    """亮度 / 对比度 / 饱和度扰动，输入输出都是 (3,H,W) 的 0~1 浮点张量。"""
    if b > 0:
        t = t * random.uniform(1 - b, 1 + b)
    if c > 0:
        m = t.mean(dim=(1, 2), keepdim=True)
        t = (t - m) * random.uniform(1 - c, 1 + c) + m
    if s > 0:
        g = (t * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
        t = (t - g) * random.uniform(1 - s, 1 + s) + g
    return t.clamp(0, 1)


class PetDataset(torch.utils.data.Dataset):
    """取图 → 增强 → 归一化 → Tensor(C,H,W)。

    aug 是增强开关的集合，可任意组合：
        ""      无增强（验证/测试也用这一档：缓存里已做过 resize+center crop，和训练的几何一致）
        "flip"  随机水平翻转
        "crop"  随机裁剪 + 缩放（scale 0.5~1.0，长宽比 3/4~4/3）
        "color" 亮度/对比度/饱和度扰动
    """

    def __init__(self, arr, idx, labels, train=False, aug="",
                 mean=(.5, .5, .5), std=(.5, .5, .5)):
        self.arr, self.idx, self.labels, self.train = arr, idx, labels, train
        self.aug = tuple(a for a in aug.split(",") if a)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        x = np.array(self.arr[self.idx[k]], copy=True)            # 内存拷贝，不解码
        if self.train and "crop" in self.aug:
            x = random_resized_crop(x)
        if self.train and "flip" in self.aug and random.random() < 0.5:
            x = x[:, ::-1]
        x = np.array(x, copy=True)                     # 保证可写：PIL 转出来的数组是只读的，转 Tensor 会告警
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div_(255.0)
        if self.train and "color" in self.aug:
            t = color_jitter(t)
        t = (t - self.mean) / self.std
        return t, int(self.labels[k])


def make_loader(split, aug="", train=False, batch=64, workers=6):
    arr = load_arr()
    sp = build_splits()
    idx = sp[split]
    labels = get_labels(idx)
    norm = compute_norm()
    ds = PetDataset(arr, idx, labels, train=train, aug=aug,
                    mean=norm["mean"], std=norm["std"])
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=workers,
        drop_last=train, persistent_workers=workers > 0)
    return loader, idx, labels


def stage_preprocess():
    section("任务 2：数据读取与预处理流程")
    norm = compute_norm()
    print("  流程：JPEG → PIL 解码 → convert('RGB') → 短边缩到 146（只缩小不放大）→ 中心裁 128×128")
    print("        → uint8 缓存 → 训练时 [随机裁剪缩放 / 水平翻转 / 颜色扰动] → float32 除 255")
    print("        → (H,W,C) 换轴成 (C,H,W) → 减均值除方差 → 堆成 (N,C,H,W) 的 batch")
    print(f"  归一化统计（训练集）：mean={[round(v, 3) for v in norm['mean']]}，"
          f"std={[round(v, 3) for v in norm['std']]}")

    loader, idx, labels = make_loader("train", aug="", train=True, batch=64, workers=0)
    xb, yb = next(iter(loader))
    print("\n  一个 batch 长什么样：")
    print(f"    xb.shape = {tuple(xb.shape)}  dtype = {xb.dtype}　"
          f"（N=64 张，C=3 通道，H=W=128 像素）")
    print(f"    yb.shape = {tuple(yb.shape)}  dtype = {yb.dtype}　"
          f"取值 {sorted(set(yb.tolist()))}（0=猫 1=狗）")
    print(f"    xb 数值范围 = {xb.min():.3f} ~ {xb.max():.3f}，均值 {xb.mean():.3f}")
    print(f"    DataLoader 做的事：按索引取图 → 逐张预处理 → collate 用 torch.stack 把 64 张")
    print(f"      形状 (3,128,128) 的图叠成 (64,3,128,128)；shuffle=True 时每个 epoch 顺序都不同。")
    print(f"    为什么这样组 batch：同尺寸才能 stack 成张量（所以要先统一到 128×128）；")
    print(f"      64 张一起算，卷积/矩阵乘的并行度高，比一张一张算快得多；")
    print(f"      取 64 是因为它同时兼顾显存/内存占用与梯度噪声（本机是 CPU，64 够快也不吃内存）。")

    meta = load_meta()
    ok = [r for r in meta if r["ok"] == "1"]
    wide = max(ok, key=lambda r: int(r["orig_w"]) / int(r["orig_h"]))
    tall = min(ok, key=lambda r: int(r["orig_w"]) / int(r["orig_h"]))
    gray = next(r for r in ok if r["mode"] == "L" and int(r["orig_w"]) > 300)
    arr = load_arr()
    chosen = [wide, tall, gray]

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(3, 5, hspace=0.5, wspace=0.2)
    for i, r in enumerate(chosen):
        row = int(r["row"])
        w0, h0 = int(r["orig_w"]), int(r["orig_h"])
        orig = Image.open(_os.path.join(PET_DIR, r["cls"], r["name"])).convert("RGB")
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(orig)
        ax.axis("off")
        ax.set_title(f"原图 {w0}×{h0}（{r['mode']}，{CLASS_CN[r['cls']]}）", fontsize=9)
        s = CACHE_SIDE / min(w0, h0)
        mid = orig.resize((max(CACHE_SIDE, int(round(w0 * s))),
                           max(CACHE_SIDE, int(round(h0 * s)))), Image.BILINEAR)
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(mid)
        ax.axis("off")
        ax.set_title(f"短边缩到 146 → {mid.size[0]}×{mid.size[1]}\n（比例没变，无拉伸）", fontsize=9)
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(arr[row])
        ax.axis("off")
        ax.set_title(f"中心裁 128×128\n（缓存里的样子）", fontsize=9)
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(arr[row])
        ax.axis("off")
        ax.set_title("放大看：有没有剪切错误\n（主体是否被裁掉/变形）", fontsize=9)
        ax = fig.add_subplot(gs[i, 4])
        x = torch.from_numpy(np.array(arr[row], copy=True)).permute(2, 0, 1).float() / 255
        me = torch.tensor(norm["mean"]).view(3, 1, 1)
        sd = torch.tensor(norm["std"]).view(3, 1, 1)
        vis = ((x - me) / sd * sd + me).clamp(0, 1)
        ax.imshow(vis.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title("归一化→反归一化\n（与左图一致 = 颜色没丢）", fontsize=9)
    fig.suptitle("任务 2：预处理前后对比 —— 检查拉伸 / 裁剪错误 / 颜色异常", fontsize=15)
    fig.savefig(_os.path.join(HERE, "06_preprocess.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("\n  抽查的三张（最扁 / 最瘦 / 灰度）：")
    for r in chosen:
        print(f"    {r['cls']}/{r['name']}：原图 {r['orig_w']}×{r['orig_h']}（{r['mode']}）"
              f" → 短边缩放 → 中心裁 128×128")
    print("  结论：全程只做“等比缩放 + 中心裁剪”，没有拉伸；统一 convert('RGB') 之后")
    print("    57 张调色板图 / 5 张灰度图 / 3 张 CMYK / 2 张 RGBA 也不会出现颜色异常；")
    print(f"    代价是极端长宽比的原图（例如最扁的那张 {wide['orig_w']}×{wide['orig_h']}）"
          "会被裁掉一部分内容。")
    print("  已保存 06_preprocess.png")


# ==================================================================
# 任务 3：搭建 CNN
# ------------------------------------------------------------------
# 结构：4 个“卷积块”＋ 全连接头
#   每个卷积块 = 卷积层(3×3) → 批归一化 → ReLU 激活 → 最大池化
#   卷积层负责“在局部找特征”，ReLU 提供非线性，池化负责“压缩尺寸、扩大感受野”
#   第一个卷积用 stride=2，相当于一进来就降一半分辨率，CPU 上省一半算力
#   最后的 1×1 全连接头把 256×4×4 的特征压成 2 个分数（猫 / 狗）
# ==================================================================
def conv_bn(i, o, stride=1):
    """卷积 + 批归一化 + ReLU，是 CNN 里最小的重复单元。"""
    return nn.Sequential(
        nn.Conv2d(i, o, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True))


class CatDogCNN(nn.Module):
    def __init__(self, width=32, dropout=0.0, nclass=2):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn(3, w, stride=2),              # 0: 128 -> 64，通道 3 -> w
            conv_bn(w, w),                        # 1
            nn.MaxPool2d(2),                      # 2: 64 -> 32
            conv_bn(w, w * 2),                    # 3
            conv_bn(w * 2, w * 2),                # 4
            nn.MaxPool2d(2),                      # 5: 32 -> 16
            conv_bn(w * 2, w * 4),                # 6
            conv_bn(w * 4, w * 4),                # 7
            nn.MaxPool2d(2),                      # 8: 16 -> 8
            conv_bn(w * 4, w * 8),                # 9: 最后一层卷积，8×8×(8w)
            nn.MaxPool2d(2),                      # 10: 8 -> 4
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


# ------------------------------------------------------------------
# 对照组：不用卷积、直接堆全连接层的“朴素”做法（思考题 1 用）
# ------------------------------------------------------------------
class MLPBaseline(nn.Module):
    """把 (3,128,128) 直接铺平成 49152 维向量，接一层巨大的全连接层。

    这是"不利用图像结构"的极端做法：每个输入像素到每个隐藏单元都有一条独立权重，
    既没有局部连接、也没有权重共享，因此参数量 = 49152 × hidden。
    hidden=512 时是 25.2 M 参数（CNN 的 23 倍）；hidden=22 时约 1.08 M，正好和 CNN 参数量相当。
    """

    def __init__(self, hidden=512, dropout=0.0, nclass=2, in_size=IMG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(3 * in_size * in_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, nclass),
        )

    def forward(self, x):
        return self.net(x)


def build_model(kind="cnn", width=32, hidden=512, dropout=0.0, nclass=2):
    if kind == "mlp":
        return MLPBaseline(hidden=hidden, dropout=dropout, nclass=nclass)
    return CatDogCNN(width=width, dropout=dropout, nclass=nclass)


def stage_arch():
    section("任务 3 / 4 / 9：CNN 结构、各层 Tensor 形状、模型结构图")
    set_seed()
    model = CatDogCNN(width=32)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{n_par / 1e6:.3f} M（{n_par} 个）")
    print(f"  其中卷积层 {sum(p.numel() for p in model.features.parameters()) / 1e6:.3f} M，"
          f"全连接头 {sum(p.numel() for p in model.head.parameters()) / 1e6:.3f} M")
    print("  逐层结构：")
    n_layers = 0
    for name, m in model.features.named_children():
        if isinstance(m, nn.Sequential):
            c = m[0]
            print(f"    features[{name}]  Conv2d(in={c.in_channels}, out={c.out_channels}, "
                  f"k=3, s={c.stride[0]}, p=1, bias=False) + BatchNorm2d + ReLU")
        else:
            print(f"    features[{name}]  {type(m).__name__}(kernel_size=2, stride=2)")
        n_layers += 1
    print(f"    head         Flatten → Dropout → Linear({32 * 8 * 4 * 4}, 128) → ReLU → Linear(128, 2)")

    # ---------- 任务 4：形状怎么变 ----------
    print("\n  任务 4：一张 (1,3,128,128) 的图片过每一层之后的形状")
    x = torch.randn(1, 3, IMG, IMG)
    shapes = [("输入图片", tuple(x.shape), "—")]
    hooks = []

    def make_hook(i, name):
        def hook(module, inp, out):
            shapes.append((name, tuple(out.shape), ""))
        return hook

    names = {0: "Conv3×3 s2 (w=32)", 1: "Conv3×3 (w=32)", 2: "MaxPool2×2",
             3: "Conv3×3 (w=64)", 4: "Conv3×3 (w=64)", 5: "MaxPool2×2",
             6: "Conv3×3 (w=128)", 7: "Conv3×3 (w=128)", 8: "MaxPool2×2",
             9: "Conv3×3 (w=256)", 10: "MaxPool2×2"}
    for i, m in model.features.named_children():
        hooks.append(m.register_forward_hook(make_hook(int(i), names[int(i)])))
    with torch.no_grad():
        logits = model(x)
    for h in hooks:
        h.remove()
    shapes.append(("Flatten", (1, 32 * 8 * 4 * 4), "铺平"))
    shapes.append(("Linear(4096→128)+ReLU", (1, 128), "特征向量"))
    shapes.append(("Linear(128→2)", tuple(logits.shape), "logits（猫/狗分数）"))

    prev = None
    for name, shp, note in shapes:
        if prev is None:
            change = ""
        else:
            dw = f"{shp[-1] / prev[-1]:.2g}×" if len(shp) > 2 and len(prev) > 2 else ""
            change = f"H,W {dw}" if dw else ""
        print(f"    {name:22s} {str(shp):18s} {change} {note}")
        prev = shp
    print("  规律：")
    print("    通道数：3 → 32 → 32 → 64 → 64 → 128 → 128 → 256 → 256（越深通道越多，"
          "每个通道记一种“特征”）")
    print("    宽高   ：128 → 64 → 32 → 16 → 8 → 4（每次池化或 stride=2 卷积砍一半），"
          "最后只剩 4×4")
    print("    总信息量：3×128×128 = 49152 个数 → 256×4×4 = 4096 个数，"
          "空间变小、语义变强，最后用全连接层压成 2 个分数")

    # 各层激活值的“样子”拼成一张图
    acts = []
    handles = []

    def grab(module, inp, out):
        acts.append(out.detach()[0])

    for _, m in model.features.named_children():
        handles.append(m.register_forward_hook(grab))
    arr = load_arr()
    sp = build_splits()
    norm = compute_norm()
    sample_row = int(sp["test"][3])
    raw = np.array(arr[sample_row], copy=True)
    xt = torch.from_numpy(raw).permute(2, 0, 1).float() / 255
    xt = ((xt - torch.tensor(norm["mean"]).view(3, 1, 1))
          / torch.tensor(norm["std"]).view(3, 1, 1)).unsqueeze(0)
    with torch.no_grad():
        model(xt)
    for h in handles:
        h.remove()

    fig = plt.figure(figsize=(19, 8.5))
    n_show = len(acts)
    for k, a in enumerate(acts):
        ax = fig.add_subplot(2, 6, k + 1)
        m = a.mean(0).numpy()
        m = (m - m.min()) / (np.ptp(m) + 1e-8)
        ax.imshow(m, cmap="viridis")
        ax.axis("off")
        t_layer = ["Conv s2", "Conv", "MaxPool", "Conv", "Conv", "MaxPool",
                   "Conv", "Conv", "MaxPool", "Conv", "MaxPool"][k]
        ax.set_title(f"第 {k} 层 {t_layer}\n{a.shape[0]} 通道 {a.shape[1]}×{a.shape[2]}",
                     fontsize=9)

    ax = fig.add_subplot(2, 6, 12)
    ax.axis("off")
    rows = [[n, str(s)] for n, s, _ in shapes[:7]]
    rows += [["…", "…"], [shapes[-2][0], str(shapes[-2][1])],
             [shapes[-1][0], str(shapes[-1][1])]]
    tbl = ax.table(cellText=rows, colLabels=["层", "输出形状"], loc="center",
                   cellLoc="left", colWidths=[0.62, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(0.95, 1.25)
    ax.set_title("各层输出形状（前 7 层 + 最后两层）", fontsize=9)
    fig.suptitle("任务 4：一张图经过 CNN 各层后的形状变化（每格是“通道求平均”后的激活图）",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(_os.path.join(HERE, "07_shapes.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 07_shapes.png")

    # ---------- 任务 9：模型结构图 ----------
    draw_arch_diagram(width=32)
    print("  已保存 12_model_arch.png")


def draw_arch_diagram(width=32):
    """用 matplotlib 画一张模型结构图（任务 9）。"""
    w = width
    blocks = [
        ("输入图片", "3×128×128", "#DDDDDD", "RGB 像素，已归一化"),
        ("Conv3×3 s2", f"{w}×64×64", "#4C72B0", "找边缘/颜色，BN+ReLU"),
        ("Conv3×3\n+Pool", f"{w}×32×32", "#5B8AC6", "局部纹理组合"),
        ("Conv3×3×2\n+Pool", f"{w * 2}×16×16", "#6E9FD0", "更大范围的纹理"),
        ("Conv3×3×2\n+Pool", f"{w * 4}×8×8", "#8AB4DA", "类部件（耳朵/眼睛/口鼻）"),
        ("Conv3×3\n+Pool", f"{w * 8}×4×4", "#A8C8E4", "整体语义"),
        ("Flatten", f"{w * 8 * 4 * 4}", "#E4C89A", "铺平成向量"),
        ("Linear→128\n+ReLU", "128", "#F0A868", "特征向量"),
        ("Linear→2", "2", "#C44E52", "logits: 猫/狗"),
    ]
    fig, ax = plt.subplots(figsize=(16.5, 4.6))
    n = len(blocks)
    bw, gap = 1.30, 0.5
    for i, (name, shape, color, note) in enumerate(blocks):
        cx = i * (bw + gap)
        ax.add_patch(FancyBboxPatch((cx, 0.35), bw, 1.05, boxstyle="round,pad=0.05",
                                    linewidth=1.2, edgecolor="#333333", facecolor=color,
                                    alpha=0.9))
        ax.text(cx + bw / 2, 1.13, name, ha="center", va="center", fontsize=9,
                color="white" if i >= 1 else "black", fontweight="bold")
        ax.text(cx + bw / 2, 0.72, shape, ha="center", va="center", fontsize=9,
                color="white" if i >= 1 else "black")
        ax.text(cx + bw / 2, 0.12, note, ha="center", va="center", fontsize=7.5, color="#333333")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((cx + bw, 0.875), (cx + bw + gap, 0.875),
                                         arrowstyle="-|>", mutation_scale=14,
                                         linewidth=1.2, color="#333333"))
    ax.set_xlim(-0.3, n * (bw + gap) - gap + 0.3)
    ax.set_ylim(-0.15, 1.7)
    ax.axis("off")
    ax.text(0.0, 1.62, "CatDogCNN 结构图（每个卷积层后接 BatchNorm + ReLU；MaxPool 的 kernel=2, stride=2）",
            fontsize=10)
    fig.tight_layout()
    fig.savefig(_os.path.join(HERE, "12_model_arch.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ==================================================================
# 任务 5：特征图可视化
# ==================================================================
def _norm_map(m):
    m = m - m.min()
    return m / (m.max() + 1e-8)


def _montage(maps, cols, gap=2, size=None):
    """把若干二维图拼成一张大图（每张先归一化到 0~1）。"""
    imgs = [_norm_map(np.asarray(m, dtype=np.float32)) for m in maps]
    if size:
        imgs = [np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((size, size),
                                                                            Image.NEAREST)) / 255.0
                for m in imgs]
    h, w = imgs[0].shape
    rows = int(math.ceil(len(imgs) / cols))
    out = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), np.float32)
    for k, m in enumerate(imgs):
        r, c = divmod(k, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = m
    return out


def stage_features(model=None, ckpt=None):
    section("任务 5：卷积特征图可视化")
    if model is None:
        model = CatDogCNN(width=32)
        ckpt = ckpt or _os.path.join(RUN_DIR, "final.pth")
        if _os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(sd["state"] if "state" in sd else sd)
            print(f"  已载入 {ckpt}")
        else:
            print("  没有训练好的权重，用随机初始化的网络演示（图的样子会不对，"
                  "真正的结论请看 note.md 里配合训练后模型的结果）")
    model.eval()

    arr = load_arr()
    sp = build_splits()
    norm = compute_norm()
    te = sp["test"]
    picks = [int(te[0]), int(te[1]), int(te[2]), int(te[3])]
    labels = get_labels(te)
    label_of = {int(r): int(l) for r, l in zip(te, labels)}
    meta = load_meta()

    first, last = {}, {}
    h1 = model.features[0].register_forward_hook(
        lambda m, i, o: first.__setitem__("out", o.detach()))
    h2 = model.features[9].register_forward_hook(
        lambda m, i, o: last.__setitem__("out", o.detach()))

    means = torch.tensor(norm["mean"]).view(3, 1, 1)
    stds = torch.tensor(norm["std"]).view(3, 1, 1)
    xs, f1s, f9s, preds = [], [], [], []
    for r in picks:
        x = torch.from_numpy(np.array(arr[r], copy=True)).permute(2, 0, 1).float() / 255
        x = ((x - means) / stds).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        xs.append((x[0] * stds + means).clamp(0, 1))
        f1s.append(first["out"][0])
        f9s.append(last["out"][0])
        preds.append(int(out.argmax(1)))
    h1.remove()
    h2.remove()

    # 第一层 32 个卷积核本身长什么样（每个 3×3×3，直接按 RGB 显示）
    k = model.features[0][0].weight.detach().numpy()           # (32,3,3,3)
    k = (k - k.min()) / (k.max() - k.min() + 1e-8)
    kh = kw = 3
    cell, gap = 3, 2
    kernel_montage = np.ones((8 * (kh + gap), 4 * (kw + 4), 3), np.float32)
    for i in range(32):
        r, c = divmod(i, 4)
        y0, x0 = r * (kh + gap), c * (kw + 4)
        kernel_montage[y0:y0 + kh, x0:x0 + kw] = np.transpose(k[i], (1, 2, 0))

    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(5, 8, hspace=0.42, wspace=0.18)
    for i, r in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(np.asarray(xs[i].permute(1, 2, 0)))
        ax.axis("off")
        m = meta[r]
        ok = "（预测对）" if preds[i] == label_of[r] else "（预测错）"
        ax.set_title(f"测试图 {i + 1}：{CLASS_CN[CLASSES[label_of[r]]]}{ok}\n"
                     f"{m['cls']}/{m['name']}  {m['orig_w']}×{m['orig_h']}", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 4:8])
    ax.imshow(kernel_montage)
    ax.axis("off")
    ax.set_title("训练后第一层的 32 个卷积核（每个 3×3×3，按 RGB 显示）\n"
                 "能看到竖直 / 水平 / 斜向条纹和渐变 —— 就是“边缘探测器”的原型；\n"
                 "部分卷积核三个通道不一样 → 专门对某种颜色差敏感", fontsize=9)

    for i in range(2):
        for j in range(8):
            ax = fig.add_subplot(gs[1 + i, j])
            ax.imshow(_norm_map(f1s[i][j].numpy()), cmap="inferno")
            ax.axis("off")
            ax.set_title(f"图{i + 1} 通道{j + 1}", fontsize=8)

    for i in range(2):
        for j in range(8):
            ax = fig.add_subplot(gs[3 + i, j])
            ax.imshow(_norm_map(f9s[i][j].numpy()), cmap="inferno")
            ax.axis("off")
            ax.set_title(f"图{i + 1} 通道{j + 1}", fontsize=8)

    fig.text(0.012, 0.63, "第一层卷积\n32 通道\n64×64\n（边缘/颜色）", fontsize=10,
             ha="left", va="center", color="#8B0000")
    fig.text(0.012, 0.28, "最后一层卷积\n256 通道\n8×8\n（语义）", fontsize=10,
             ha="left", va="center", color="#8B0000")
    fig.suptitle("任务 5：第一层 / 最后一层卷积的输出特征图（各取前 8 个通道）", fontsize=15)
    fig.savefig(_os.path.join(HERE, "08_feature_maps.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 定量：通道之间的“分工”
    act = f1s[0].abs().mean(dim=(1, 2)).numpy()
    print("\n  图 1 在第一层各通道上的平均响应（越大说明这个通道越“被激活”）：")
    for j in range(8):
        print(f"    通道 {j:2d}：{act[j]:.3f}", end="   ")
        if j % 4 == 3:
            print()
    print(f"  最强通道 {int(act.argmax())}（{act.max():.3f}），最弱通道 {int(act.argmin())}"
          f"（{act.min():.3f}），相差 {act.max() / max(act.min(), 1e-6):.0f} 倍")

    conc = []
    for c in range(act.shape[0]):
        m = f1s[0][c].abs().numpy()
        conc.append((m.max() ** 2) / (m.mean() ** 2 + 1e-8))
    conc = np.array(conc)
    print(f"  空间集中度（峰值²/均值²）中位数 {np.median(conc):.1f}")
    print(f"    最集中的 5 个通道：{np.argsort(-conc)[:5].tolist()} —— 只在图上某一小块亮，"
          "像“某个方向的边缘/角点检测器”")
    print(f"    最分散的 5 个通道：{np.argsort(conc)[:5].tolist()} —— 全图都在亮，"
          "更像“亮度/颜色偏置”通道")
    # 同一通道在不同图片上的表现
    print("\n  同一组通道在 4 张测试图上的平均响应（每行一个通道）：")
    resp = np.stack([x.abs().mean(dim=(1, 2)).numpy() for x in f1s])      # (4 图, 32 通道)
    for c in np.argsort(-resp.mean(0))[:4]:
        print("    通道 {0:2d}：".format(int(c)) +
              "，".join(f"图{i + 1} {resp[i, c]:.3f}" for i in range(4)))
    print("    可见同一个通道对不同图片的响应强度也不同 —— 它只对“自己认的那种纹理”有反应。")
    print("  已保存 08_feature_maps.png")


# ==================================================================
# 任务 6：数据增强
# ==================================================================
def stage_augment():
    section("任务 6：数据增强的对照实验（先看增强出来的样本）")
    set_seed()
    arr = load_arr()
    sp = build_splits()
    norm = compute_norm()
    row = int(sp["train"][7])
    base = np.array(arr[row], copy=True)
    meta = load_meta()
    print(f"  用训练集里的 {meta[row]['cls']}/{meta[row]['name']} 做样例")
    print("  三种增强：")
    print("    水平翻转 —— 猫狗左右翻过来还是猫狗，标签语义不变，安全（除非有文字/方向信息）")
    print("    随机裁剪缩放 —— 模拟“主体在画面中的位置/大小不同”，安全；但要注意"
          "scale 不能太小，否则可能把主体裁掉、只剩背景，标签就错了")
    print("    颜色扰动 —— 模拟不同光照/相机白平衡，安全；但幅度过大（比如把橘猫调成蓝猫）"
          "会让图像失真、偏离真实分布")

    def to_disp(x):
        return x.numpy() if isinstance(x, np.ndarray) else x

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(4, 6, hspace=0.45, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(base)
    ax.axis("off")
    ax.set_title("原始（缓存里的 128×128）", fontsize=9)

    random.seed(1)
    for j in range(5):                                          # 水平翻转 5 张
        ax = fig.add_subplot(gs[0, 1 + j])
        s = random.random() < 0.5
        ax.imshow(base[:, ::-1] if s else base)
        ax.axis("off")
        ax.set_title(f"水平翻转：{'翻了' if s else '没翻'}", fontsize=9)

    for j in range(6):                                          # 随机裁剪缩放（不同 scale）
        ax = fig.add_subplot(gs[1, j])
        random.seed(100 + j)
        ax.imshow(random_resized_crop(base))
        ax.axis("off")
        ax.set_title(f"随机裁剪缩放 #{j + 1}", fontsize=9)

    for j in range(6):                                          # 颜色扰动（不同幅度）
        ax = fig.add_subplot(gs[2, j])
        random.seed(200 + j)
        t = torch.from_numpy(base).permute(2, 0, 1).float() / 255
        amp = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2][j]
        t = color_jitter(t, amp, amp, amp)
        ax.imshow(t.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"颜色扰动 幅度{amp}", fontsize=9)

    # 危险示范：裁得太狠 / 颜色调太狠
    ax = fig.add_subplot(gs[3, 0:2])
    random.seed(9)
    ax.imshow(random_resized_crop(base, scale=(0.06, 0.12)))     # 只裁到一小块
    ax.axis("off")
    ax.set_title("危险：scale 太小（0.06~0.12）\n只剩一块背景/毛发，标签可能已经不成立", fontsize=9)
    ax = fig.add_subplot(gs[3, 2:4])
    random.seed(10)
    t = torch.from_numpy(base).permute(2, 0, 1).float() / 255
    t = color_jitter(t, 0.9, 0.9, 0.9)
    ax.imshow(t.permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title("危险：颜色幅度 0.9\n颜色已经明显失真（不像真实拍摄的照片）", fontsize=9)
    ax = fig.add_subplot(gs[3, 4:6])
    random.seed(11)
    ax.imshow(base[::-1, ::-1])                                  # 旋转 180°（这里只示范竖直翻转的等价效果）
    ax.axis("off")
    ax.set_title("危险：竖直翻转 / 大角度旋转\n猫狗倒过来就不是自然照片了；"
                 "而且如果有“天空在上”这类先验，模型会被带偏", fontsize=9)
    fig.suptitle("任务 6：数据增强样例（上面两行是合理的，下面一行是可能破坏标签/让图片失真的做法）",
                 fontsize=14)
    fig.savefig(_os.path.join(HERE, "09_augmentation.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 09_augmentation.png")
    print("  注：真正好不好用，要看下面训练出来的对照实验（无增强 vs 有增强的验证集准确率）。")


# ==================================================================
# 训练
# ==================================================================
def run_epoch(model, loader, opt, crit, train, quiet=False, channels_last=False):
    model.train(train)
    tot_loss, tot_correct, n, t0 = 0.0, 0, 0, time.time()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi, (xb, yb) in enumerate(loader):
            if channels_last:                                    # CPU 上比 NCHW 再快 ~20%
                xb = xb.contiguous(memory_format=torch.channels_last)
            if train:
                opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            if train:
                loss.backward()
                opt.step()
            tot_loss += loss.item() * yb.size(0)
            tot_correct += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
            if not quiet and train and bi % 200 == 199:
                print(f"      batch {bi + 1}/{len(loader)}  loss={tot_loss / n:.4f}  "
                      f"acc={tot_correct / n:.4f}  ({time.time() - t0:.0f}s)")
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def predict_all(model, loader, tta=False):
    """返回 (loss, acc, probs(N,2), labels(N), idx 顺序与 loader 一致)。"""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    probs, labels, tot_loss, n = [], [], 0.0, 0
    for xb, yb in loader:
        out = model(xb)
        if tta:
            out = (F.softmax(out, 1) + F.softmax(model(torch.flip(xb, dims=[3])), 1)) / 2
            loss = F.nll_loss(torch.log(out.clamp_min(1e-12)), yb, reduction="sum")
            p = out
        else:
            p = F.softmax(out, 1)
            loss = crit(out, yb)
        tot_loss += loss.item()
        probs.append(p)
        labels.append(yb)
        n += yb.size(0)
    return tot_loss / n, None, torch.cat(probs).numpy(), torch.cat(labels).numpy()


def save_checkpoint(path, model, args, norm, hist, extra=None):
    obj = {"state": model.state_dict(),
           "arch": {"kind": args.arch, "width": args.width, "hidden": args.hidden,
                    "dropout": args.dropout},
           "img": IMG, "classes": CLASSES, "norm": norm,
           "args": {k: v for k, v in vars(args).items()},
           "history": hist}
    if extra:
        obj.update(extra)
    last = None
    for _ in range(3):                       # 偶发文件被占用（杀毒/索引）时重试，别让训练白跑
        try:
            torch.save(obj, path)
            return True
        except Exception as e:               # noqa: BLE001
            last = e
            time.sleep(1.5)
    print(f"    [警告] 保存 {path} 失败：{last}")
    return False


def stage_train(args):
    if args.aug.strip().lower() in ("", "none", "no"):
        args.aug = ""                                            # 从命令行传空串容易丢，用 none 代替
    section(f"训练：tag={args.tag}  aug='{args.aug}'  epochs={args.epochs}  "
            f"dropout={args.dropout}  label_smoothing={args.ls}  subset={args.subset}\n"
            f"       模型 arch={args.arch}"
            + (f"（隐藏层 {args.hidden} 维）" if args.arch == "mlp" else f"（width={args.width}）"))
    _os.makedirs(RUN_DIR, exist_ok=True)
    t_start = time.time()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    norm = compute_norm()

    arr = load_arr()
    sp = build_splits()
    tr_idx, tr_y = sp["train"], get_labels(sp["train"])
    va_idx, va_y = sp["val"], get_labels(sp["val"])
    if args.subset and args.subset < len(tr_idx):
        keep = np.concatenate([np.where(tr_y == c)[0][:args.subset // 2] for c in range(2)])
        tr_idx, tr_y = tr_idx[keep], tr_y[keep]
        print(f"  只用了训练集的 {len(keep)} 张（每类 {args.subset // 2} 张）做对照实验")
    if args.valsub and args.valsub < len(va_idx):
        keep = np.concatenate([np.where(va_y == c)[0][:args.valsub // 2] for c in range(2)])
        va_idx, va_y = va_idx[keep], va_y[keep]
        print(f"  验证集也只用 {len(keep)} 张（每类 {args.valsub // 2} 张），只为省时间")

    def loader_of(idx, y, train, aug, workers):
        ds = PetDataset(arr, idx, y, train=train, aug=aug,
                        mean=norm["mean"], std=norm["std"])
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch, shuffle=train, num_workers=workers,
            drop_last=train, persistent_workers=workers > 0)

    tr_loader = loader_of(tr_idx, tr_y, True, args.aug, args.workers)
    # 验证集只有读操作、总量又小（几十个 batch），直接在主进程里加载，
    # 避免在 Windows 上反复 spawn 子进程（实测会偶发 OSError: Invalid argument）
    va_loader = loader_of(va_idx, va_y, False, "", 0)

    print(f"  训练集 {len(tr_idx)} 张，验证集 {len(va_idx)} 张，"
          f"batch={args.batch}，num_workers={args.workers}，torch 线程数 {args.threads}")
    model = build_model(kind=args.arch, width=args.width, hidden=args.hidden,
                        dropout=args.dropout)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  参数量 {n_par / 1e6:.3f} M")
    if n_par > 20e6:
        print(f"  （参数量超过 20 M，权重文件会超过 100 MB，只记录训练曲线、不保存权重）")
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

    hist = {"tag": args.tag, "args": vars(args), "epochs": []}
    best_acc, best_ep = -1.0, -1
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, tr_loader, opt, crit, train=True,
                                    channels_last=args.channels_last)
        va_loss, va_acc = run_epoch(model, va_loader, opt, crit, train=False, quiet=True,
                                    channels_last=args.channels_last)
        lr_now = opt.param_groups[0]["lr"]
        sched.step()
        hist["epochs"].append(dict(epoch=ep, tr_loss=tr_loss, tr_acc=tr_acc,
                                   va_loss=va_loss, va_acc=va_acc, lr=lr_now,
                                   sec=round(time.time() - t0, 1)))
        flag = ""
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, ep
            if n_par <= 20e6:
                save_checkpoint(_os.path.join(RUN_DIR, f"{args.tag}.pth"), model, args, norm, hist)
                flag = " ← 目前最好，已保存权重"
            else:
                flag = " ← 目前最好（参数太多，只记曲线不存权重）"
        print(f"    epoch {ep:2d}/{args.epochs}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}  ({time.time() - t0:.1f}s){flag}")

    with open(_os.path.join(RUN_DIR, f"{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)
    print(f"  完成：最好验证集准确率 {best_acc:.4f}（第 {best_ep} 轮），"
          f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    return hist


# ==================================================================
# 汇总：训练曲线 + 对照实验
# ==================================================================
def _load_run(tag):
    p = _os.path.join(RUN_DIR, f"{tag}.json")
    if not _os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stage_report():
    section("汇总：训练曲线与对照实验")
    names = {"noaug": "无增强", "aug": "有增强（翻转+裁剪+颜色）",
             "final": "改进版（有增强+Dropout+标签平滑）",
             "sub8_noaug": "小集6000 无增强 8 轮", "sub8_aug": "小集6000 有增强 8 轮"}
    colors = {"noaug": "#C44E52", "aug": "#4C72B0", "final": "#55A868",
              "sub8_noaug": "#937860", "sub8_aug": "#8172B3"}
    runs = {t: _load_run(t) for t in names}
    have = {t: r for t, r in runs.items() if r}
    if not have:
        print("  还没有训练记录，先跑 --stages train")
        return

    # 消融只保留“无增强 vs 单一增强”这四组（课件里点名的就是这三类增强）；
    # “三者全开”的对照由上面那张全量实验的柱状图承担
    abl_tags = ["abl_none", "abl_flip", "abl_crop", "abl_color"]
    abl_names = ["无增强", "仅水平翻转", "仅随机裁剪", "仅颜色扰动"]
    abl = [_load_run(t) for t in abl_tags]

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    for tag, name in names.items():
        r = runs.get(tag)
        if not r:
            continue
        ep = [e["epoch"] for e in r["epochs"]]
        ax_loss.plot(ep, [e["tr_loss"] for e in r["epochs"]], "--", color=colors[tag], lw=1.2)
        ax_loss.plot(ep, [e["va_loss"] for e in r["epochs"]], "-", color=colors[tag], lw=2,
                     label=name)
        ax_acc.plot(ep, [e["tr_acc"] for e in r["epochs"]], "--", color=colors[tag], lw=1.2)
        ax_acc.plot(ep, [e["va_acc"] for e in r["epochs"]], "-", color=colors[tag], lw=2,
                    label=name)
    for ax, ttl, ylab in ((ax_loss, "loss（虚线=训练集，实线=验证集）", "loss"),
                          (ax_acc, "accuracy（虚线=训练集，实线=验证集）", "accuracy")):
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylab)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7, loc="best")
    ax_acc.set_ylim(0.5, 1.02)

    ax = fig.add_subplot(gs[0, 2])
    for tag, r in runs.items():
        if not r:
            continue
        ep = [e["epoch"] for e in r["epochs"]]
        gap = [e["tr_acc"] - e["va_acc"] for e in r["epochs"]]
        ax.plot(ep, gap, "-o", ms=3, color=colors[tag], label=names[tag])
    ax.set_title("过拟合差距 = 训练准确率 − 验证准确率\n（越高说明越“背答案”）", fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("差距"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0:2])
    best = [max(e["va_acc"] for e in r["epochs"]) if r else 0 for r in abl]
    bars = ax.bar(abl_names, best, color=["#C44E52", "#4C72B0", "#55A868", "#CCB974"])
    for b, v, r in zip(bars, best, abl):
        if r:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("验证集最高准确率")
    n_sub = next((r["args"].get("subset") for r in abl if r), None)
    n_ep = next((r["args"].get("epochs") for r in abl if r), None)
    ax.set_title(f"数据增强消融实验（训练集只用 {n_sub} 张、{n_ep} 轮，每项只开一种增强）",
                 fontsize=11)

    ax = fig.add_subplot(gs[1, 2])
    main_tags = [t for t in ("noaug", "aug", "final") if runs.get(t)]
    vals = [max(e["va_acc"] for e in runs[t]["epochs"]) for t in main_tags]
    short = {"noaug": "无增强\n4 轮", "aug": "有增强\n4 轮", "final": "改进版\n8 轮"}
    bars = ax.bar([short.get(t, names[t]) for t in main_tags], vals,
                  color=[colors[t] for t in main_tags])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("验证集最高准确率")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("全量训练集的主要对照", fontsize=11)

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    lines = []
    rows = []
    for tag in (("noaug", "aug", "final", "sub8_noaug", "sub8_aug")
                + tuple(t for t in abl_tags if _load_run(t))):
        r = _load_run(tag)
        if not r:
            continue
        a = r["args"]
        best_acc = max(e["va_acc"] for e in r["epochs"])
        note = f"aug={a['aug'] or '无'}"
        if a.get("dropout"):
            note += f", dp={a['dropout']}"
        if a.get("ls"):
            note += f", ls={a['ls']}"
        rows.append([tag, names.get(tag, abl_names[abl_tags.index(tag)] if tag in abl_tags else tag),
                     "全量" if not a.get("subset") else str(a["subset"]), str(a["epochs"]),
                     f"{best_acc:.4f}", note])
        lines.append(f"{tag:<10s}{rows[-1][2]:>6s} 张  {a['epochs']:>2d} 轮  "
                     f"最好验证准确率 {best_acc:.4f}   {note}")
    if rows:
        tbl = ax.table(cellText=rows,
                       colLabels=["tag", "实验", "训练集", "轮数", "最好验证准确率", "说明"],
                       loc="center", cellLoc="center",
                       colWidths=[0.13, 0.26, 0.09, 0.07, 0.16, 0.29])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.45)
    fig.suptitle("任务 6：训练曲线与数据增强对照实验", fontsize=15)
    fig.savefig(_os.path.join(HERE, "10_training_curves.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 10_training_curves.png")
    for line in lines:
        print("    " + line)


# ==================================================================
# 任务 7：错误分析
# ==================================================================
def _sharpness(img):
    g = np.asarray(img, dtype=np.float32).mean(2)
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def load_model(ckpt, threads=8, tta=False):
    torch.set_num_threads(threads)
    obj = torch.load(ckpt, map_location="cpu")
    arch = obj["arch"]
    model = build_model(kind=arch.get("kind", "cnn"), width=arch.get("width", 32),
                        hidden=arch.get("hidden", 512), dropout=arch.get("dropout", 0.0))
    model.load_state_dict(obj["state"])
    model.eval()
    return model, obj


def stage_errors(ckpt=None, aug=""):
    section("任务 7：查看预测错误的样本并分析原因")
    ckpt = ckpt or _os.path.join(RUN_DIR, "final.pth")
    if not _os.path.exists(ckpt):
        print(f"  找不到 {ckpt}，先训练")
        return
    model, obj = load_model(ckpt)
    norm = obj["norm"]
    arr = load_arr()
    meta = load_meta()
    sp = build_splits()
    te, te_y = sp["test"], get_labels(sp["test"])
    ds = PetDataset(arr, te, te_y, train=False, aug="",
                    mean=norm["mean"], std=norm["std"])
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    _, _, probs, labels = predict_all(model, loader, tta=True)
    pred = probs.argmax(1)
    acc = (pred == labels).mean()
    print(f"  测试集 {len(te)} 张，准确率（含水平翻转 TTA）= {acc:.4f}")

    cm = np.zeros((2, 2), int)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    print("  混淆矩阵（行=真实，列=预测）：")
    print(f"    真实猫 → 预测猫 {cm[0, 0]}，预测狗 {cm[0, 1]}")
    print(f"    真实狗 → 预测猫 {cm[1, 0]}，预测狗 {cm[1, 1]}")

    # 按原始尺寸 / 清晰度 / 色彩模式 分组统计
    sharp = np.array([_sharpness(arr[int(r)]) for r in te])
    size = np.array([int(meta[int(r)]["orig_w"]) * int(meta[int(r)]["orig_h"]) for r in te])
    mode = np.array([meta[int(r)]["mode"] for r in te])
    correct = (pred == labels)
    groups = [
        ("小图（<1.5 万像素）", size < 15000),
        ("大图（≥1.5 万像素）", size >= 15000),
        (f"较糊（方差<{np.median(sharp):.0f}）", sharp < np.median(sharp)),
        (f"较清晰（≥{np.median(sharp):.0f}）", sharp >= np.median(sharp)),
        ("非 RGB 原图", mode != "RGB"),
        ("RGB 原图", mode == "RGB"),
    ]
    print("  分组准确率：")
    for name, m in groups:
        if m.sum():
            print(f"    {name:<24s} {m.sum():5d} 张   准确率 {correct[m].mean():.4f}")

    # 找“最自信的错误”和“最自信的正确”
    conf = probs.max(1)
    err = np.where(~correct)[0]
    order = err[np.argsort(-conf[err])]
    top_err = order[:12]
    print(f"\n  错误共 {len(err)} 张（{100 * len(err) / len(te):.1f}%），"
          f"最自信的 12 个错误样本：")
    for k in top_err:
        m = meta[int(te[k])]
        print(f"    {m['cls']}/{m['name']:<9s} 原图 {m['orig_w']}×{m['orig_h']:<5s}"
              f" 真实={CLASS_CN[CLASSES[labels[k]]]} 预测={CLASS_CN[CLASSES[pred[k]]]}"
              f" 置信度 {conf[k]:.3f}  清晰度 {sharp[k]:.1f}")

    def show(ax, k, title_color):
        ax.imshow(arr[int(te[k])])
        ax.axis("off")
        m = meta[int(te[k])]
        ax.set_title(f"真 {CLASS_CN[CLASSES[labels[k]]]} / 预 {CLASS_CN[CLASSES[pred[k]]]} "
                     f"{conf[k]:.2f}\n{m['orig_w']}×{m['orig_h']}  清晰度{sharp[k]:.0f}",
                     fontsize=7.5, color=title_color)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(4, 6, hspace=0.45, wspace=0.15)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["预测猫", "预测狗"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["真实猫", "真实狗"])
    ax.set_title(f"混淆矩阵（准确率 {acc:.4f}）", fontsize=11)

    ax = fig.add_subplot(gs[0, 2:6])
    names = [g[0] for g in groups]
    vals = [correct[g[1]].mean() if g[1].sum() else 0 for g in groups]
    ax.barh(names[::-1], vals[::-1], color="#4C72B0")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.0)
    ax.set_title("分组准确率：模型在哪类图上更容易错", fontsize=11)

    for j, k in enumerate(top_err[:6]):
        show(fig.add_subplot(gs[1, j]), k, "#8B0000")
    for j, k in enumerate(top_err[6:12]):
        show(fig.add_subplot(gs[2, j]), k, "#8B0000")
    good = np.where(correct)[0]
    good = good[np.argsort(-conf[good])][:6]
    for j, k in enumerate(good):
        show(fig.add_subplot(gs[3, j]), k, "#006400")
    fig.text(0.0, 0.335, " ↑ 两行：最“自信地错”的 12 张     ↓ 最自信地对的 6 张",
             fontsize=11, color="#8B0000")
    fig.suptitle("任务 7：错误样本分析（含水平翻转 TTA）", fontsize=15)
    fig.savefig(_os.path.join(HERE, "11_error_analysis.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 11_error_analysis.png")

    with open(_os.path.join(RUN_DIR, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(dict(acc=float(acc), cm=cm.tolist(),
                       groups={n: [int(m.sum()), float(correct[m].mean()) if m.sum() else None]
                               for n, m in groups},
                       top_errors=[dict(name=meta[int(te[k])]["name"],
                                        cls=meta[int(te[k])]["cls"],
                                        true=int(labels[k]), pred=int(pred[k]),
                                        conf=float(conf[k]), sharp=float(sharp[k]),
                                        size=int(size[k]))
                                   for k in top_err]),
                  f, ensure_ascii=False, indent=2)
    return dict(acc=float(acc), cm=cm.tolist())


# ==================================================================
# 思考题 1：卷积层 vs 直接堆全连接层
# ==================================================================
def stage_cnn_vs_mlp():
    section("思考题 1：卷积层 vs 巨型全连接层（同样的数据、优化器、轮数）")
    cand = [("noaug", "CNN 1.108M，4 轮", "#4C72B0"),
            ("mlp22", "MLP22 1.081M，4 轮 lr=1e-3", "#55A868"),
            ("mlp512", "MLP512 25.17M，4 轮 lr=1e-3", "#C44E52"),
            ("mlp512_lr12", "MLP512 25.17M，12 轮 lr=1e-4", "#937860"),
            ("mlp22_lr12", "MLP22 1.081M，12 轮 lr=1e-4", "#CCB974")]
    runs = [(t, n, c, _load_run(t)) for t, n, c in cand]
    have = [(t, n, c, r) for t, n, c, r in runs if r]
    if not have:
        print("  还没有对照实验记录，先跑：")
        print("    --stages train --tag noaug   --arch cnn --aug none --epochs 4")
        print("    --stages train --tag mlp22   --arch mlp --hidden 22  --aug none --epochs 4")
        print("    --stages train --tag mlp512  --arch mlp --hidden 512 --aug none --epochs 4")
        return

    lines = []
    for tag, name, _, r in have:
        a = r["args"]
        best = max(e["va_acc"] for e in r["epochs"])
        secs = float(np.median([e["sec"] for e in r["epochs"]]))   # 取中位数，个别轮次被机器卡顿带偏也不受影响
        n_par = sum(p.numel() for p in build_model(
            kind=a.get("arch", "cnn"), width=a.get("width", 32),
            hidden=a.get("hidden", 512)).parameters())
        # 早期几次训练的 json 里没有 arch/lr/threads 这些字段（加对照实验之前跑的），给默认值
        lines.append((tag, a.get("arch", "cnn"), n_par, best, secs, r["epochs"][-1]["tr_acc"],
                      len(r["epochs"]), a.get("lr", 1e-3), a.get("threads", 14)))
    print(f"  {'tag':<14}{'结构':<5}{'参数量':>10}{'轮数':>5}{'lr':>8}{'最好验证':>11}"
          f"{'训练准确率':>12}{'每轮秒':>8}")
    for tag, arch, n_par, best, secs, tr_acc, n_ep, lr, th in lines:
        print(f"  {tag:<14}{arch:<5}{(n_par or 0) / 1e6:>9.3f}M{n_ep:>5d}{lr:>8.0e}"
              f"{best:>11.4f}{tr_acc:>12.4f}{secs:>8.1f}")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for tag, name, color, r in have:
        ax.plot([e["epoch"] for e in r["epochs"]], [e["va_acc"] for e in r["epochs"]],
                "-o", ms=4, color=color, label=name.replace("\n", " "))
    ax.set_title("验证集准确率", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for tag, name, color, r in have:
        ax.plot([e["epoch"] for e in r["epochs"]], [e["tr_acc"] for e in r["epochs"]],
                "-o", ms=4, color=color, label=name.replace("\n", " "))
    ax.set_title("训练集准确率（越高越容易过拟合）", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    for tag, name, color, r in have:
        gap = [e["tr_acc"] - e["va_acc"] for e in r["epochs"]]
        ax.plot([e["epoch"] for e in r["epochs"]], gap, "-o", ms=4, color=color,
                label=name.replace("\n", " "))
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("过拟合差距 = 训练准确率 − 验证准确率", fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_ylabel("差距")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(lines))
    ax.bar(xs, [L[2] / 1e6 for L in lines],
           color=[c for _, _, c, _ in have])
    ax.set_yscale("log")
    ax.set_ylim(0.5, 300)
    ax.set_xticks(xs)
    ax.set_xticklabels([L[0] for L in lines], fontsize=7.5, rotation=12)
    for x, L in zip(xs, lines):
        ax.text(x, L[2] / 1e6, f"{L[2] / 1e6:.2f}M\n验证 {L[3]:.4f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_title("参数量（对数轴）与验证准确率", fontsize=11)
    ax.set_ylabel("参数量（百万）")

    ax = fig.add_subplot(gs[1, 1])
    same = [(t, s, c) for (t, _, _, _, s, _, n_ep, _, th), (_, _, c, _) in zip(lines, have)
            if n_ep == 4]                       # 只比"同样 4 轮"的三次，机器状态一致，耗时可横向比
    ax.bar([t for t, _, c in same], [s for _, s, _ in same],
           color=[c for _, _, c in same])
    for x, (_, s, _) in enumerate(same):
        ax.text(x, s, f"{s:.0f}s", ha="center", va="bottom", fontsize=9)
    ax.set_title("每轮训练耗时中位数（只列同样 4 轮、14 线程的三次）", fontsize=10)
    ax.set_ylabel("秒")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    rows = [[t, a, f"{(n_par or 0) / 1e6:.3f}M", str(n_ep), f"{lr:.0e}", f"{best:.4f}",
             f"{tr_acc:.4f}"] for t, a, n_par, best, secs, tr_acc, n_ep, lr, th in lines]
    tbl = ax.table(cellText=rows,
                   colLabels=["tag", "结构", "参数量", "轮数", "lr", "验证准确率", "训练准确率"],
                   loc="center", cellLoc="center",
                   colWidths=[0.16, 0.09, 0.13, 0.08, 0.1, 0.16, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.5)
    fig.suptitle("思考题 1：卷积层 vs 把图片铺平后接巨型全连接层（全量训练集、无增强；"
                 "前两组同为 4 轮 / lr=1e-3，后两组是给 MLP 调小学习率并训练 12 轮）", fontsize=13)
    fig.savefig(_os.path.join(HERE, "14_cnn_vs_mlp.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  已保存 14_cnn_vs_mlp.png")


# ==================================================================
# 对比：若干模型在测试集上的表现（任务 7 用）
# ==================================================================
def stage_compare(tags=("noaug", "final"), threads=12):
    section("对比：不同模型在测试集上的表现")
    arr = load_arr()
    sp = build_splits()
    te, te_y = sp["test"], get_labels(sp["test"])
    out = {}
    preds = {}
    for tag in tags:
        ckpt = _os.path.join(RUN_DIR, f"{tag}.pth")
        if not _os.path.exists(ckpt):
            print(f"  跳过 {tag}（没有 {ckpt}）")
            continue
        model, obj = load_model(ckpt, threads=threads)
        ds = PetDataset(arr, te, te_y, train=False, aug="",
                        mean=obj["norm"]["mean"], std=obj["norm"]["std"])
        loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        row = {}
        for tta in (False, True):
            _, _, p, lab = predict_all(model, loader, tta=tta)
            acc = float((p.argmax(1) == lab).mean())
            row["acc" + ("_tta" if tta else "")] = acc
            if tta:
                preds[tag] = p.argmax(1)
                row["probs"] = p
        row["params"] = sum(q.numel() for q in model.parameters())
        row["val_best"] = max(e["va_acc"] for e in obj["history"]["epochs"])
        row["epochs"] = obj["args"]["epochs"]
        row["aug"] = obj["args"]["aug"]
        out[tag] = row
        print(f"  {tag:8s} 验证集 {row['val_best']:.4f} | 测试集 {row['acc']:.4f}"
              f" | 测试集+TTA {row['acc_tta']:.4f} | 参数 {row['params'] / 1e6:.3f}M"
              f" | aug='{row['aug']}' {row['epochs']} 轮")

    if len(preds) >= 2:
        ks = list(preds)
        a, b = ks[0], ks[-1]
        pa, pb = preds[a] == te_y, preds[b] == te_y
        print(f"\n  以 {b} 与 {a} 对比（都在测试集上，含 TTA）：")
        print(f"    两者都错：{int((~pa & ~pb).sum())} 张")
        print(f"    {a} 错、{b} 对：{int((~pa & pb).sum())} 张（改进版修好的）")
        print(f"    {a} 对、{b} 错：{int((pa & ~pb).sum())} 张（改进版新弄错的）")
        out["compare"] = {f"{a}_wrong": int((~pa).sum()), f"{b}_wrong": int((~pb).sum()),
                          "both_wrong": int((~pa & ~pb).sum()),
                          "fixed": int((~pa & pb).sum()), "broken": int((pa & ~pb).sum())}
    for k in list(out):
        if isinstance(out[k], dict) and "probs" in out[k]:
            out[k] = {kk: vv for kk, vv in out[k].items() if kk != "probs"}
    with open(_os.path.join(RUN_DIR, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("  已保存 runs/compare.json")


# ==================================================================
# 命令行入口
# ==================================================================
ALL_STAGES = ["cache", "explore", "preprocess", "arch", "features", "augment",
              "train", "report", "errors", "compare", "cnnmlp"]


def main():
    ap = argparse.ArgumentParser(
        description="Task 2 · 实践 2：训练猫狗分类器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["all"], choices=ALL_STAGES + ["all"],
                    help="要跑哪几个阶段（可以写多个）")
    ap.add_argument("--tag", default="run", help="本次训练的名字（决定 runs/<tag>.json / .pth）")
    ap.add_argument("--aug", default="", help="增强开关，逗号分隔：flip,crop,color（空=无增强）")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ls", type=float, default=0.0, help="CrossEntropyLoss 的 label_smoothing")
    ap.add_argument("--width", type=int, default=32, help="第一个卷积块的通道数")
    ap.add_argument("--arch", default="cnn", choices=["cnn", "mlp"],
                    help="cnn=卷积网络；mlp=直接铺平后接全连接层（思考题 1 的对照）")
    ap.add_argument("--hidden", type=int, default=512, help="arch=mlp 时隐藏层的宽度")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8, help="torch 的 CPU 线程数")
    ap.add_argument("--subset", type=int, default=0, help="只用训练集的前 N 张（0=全用）")
    ap.add_argument("--valsub", type=int, default=0, help="验证集只用前 N 张（0=全用）")
    ap.add_argument("--channels-last", action="store_true", default=True,
                    help="CPU 上用 channels_last 内存格式，卷积快约 20%%")
    ap.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ckpt", default=None, help="errors/features 阶段用的权重文件")
    args = ap.parse_args()

    stages = ALL_STAGES if "all" in args.stages else args.stages
    torch.set_num_threads(args.threads)
    t0 = time.time()
    for st in stages:
        if st == "cache":
            build_cache()
        elif st == "explore":
            stage_explore()
        elif st == "preprocess":
            stage_preprocess()
        elif st == "arch":
            stage_arch()
        elif st == "features":
            stage_features(ckpt=args.ckpt)
        elif st == "augment":
            stage_augment()
        elif st == "train":
            stage_train(args)
        elif st == "report":
            stage_report()
        elif st == "errors":
            stage_errors(ckpt=args.ckpt)
        elif st == "compare":
            stage_compare(threads=args.threads)
        elif st == "cnnmlp":
            stage_cnn_vs_mlp()
    print(f"\n全部完成，总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()





