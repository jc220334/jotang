# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 4 · 拓展 3：Transformer 跑到图像上 —— ViT（Vision Transformer）

题目（拓展 3）：
    Transformer 后来不只被用于文字。选择一个你感兴趣的方向（图像 / 语音 / 多模态），
    找出一个使用 Transformer 的模型（例如 ViT），简单说明：
        · 它的输入是什么；
        · Attention 在哪里发挥作用；
        · 它和原论文有什么不同。

本脚本做的事（把上面三问用一张图说清楚）：
    * 拿一张真实的猫图，切成 16×16 的小块（patch），展示"图像 → 序列"这一步：
      224×224 的图变成 14×14 = 196 个 patch，每个 patch 展平是 16·16·3 = 768 维；
    * 把每个 patch 线性投影成 token（ViT 里这个投影矩阵是学出来的，这里用随机正交投影，
      只为了说清形状和"patch 之间可以互相注意"这件事）；
    * 用 Task 4 里手写的 Scaled Dot-Product Attention 算 patch 之间的注意力，
      画出"某个 patch 在看哪些 patch"，并叠加回原图；
    * 打印 token 数、注意力矩阵大小，并和 CNN 的局部感受野做对比；
    * 文字部分：ViT 的输入 / 训练方式 / 与原论文的三点不同。

注意（保持诚实）：这里**没有加载预训练权重**，注意力只是一个带温度的点积相似度，
所以"它关注到哪里"不代表真实 ViT 学到的东西，只用来展示"图像问题怎么变成序列问题"。
真实 ViT 的注意力图需要训练或加载 checkpoint 才能解释（论文附录里有大量例子）。

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python vit_patch_demo.py
    2) conda run -n jotang-ml python vit_patch_demo.py
    3) python vit_patch_demo.py --image "..\\Jotang-ml-task2\\data\\PetImages\\Cat\\0.jpg" --patch-row 5 --patch-col 7

输出：06_vit_patches.png
"""

# ==================================================================
# 【第 0 步】运行环境自举（关键，不要删）
# ------------------------------------------------------------------
# 现象：不激活 conda 环境而直接运行 python.exe 时，进程做矩阵乘（np.dot / @）
#       的一瞬间会原生崩溃，报 "Windows fatal exception: code 0xc06d007f"。
# 原因：本环境 numpy 的 MKL 在 conda 环境的 Library\bin 下；不激活环境时该目录
#       不在 PATH 上，numpy 会载入错误的 BLAS，而 matplotlib 保存图片内部要用 np.dot。
# 解决：在 import numpy 之前，把当前环境的 DLL 目录补进搜索路径。
# ==================================================================
import os as _os
import sys as _sys

_ENV_ROOT = _os.path.dirname(_sys.executable)          # 例如 D:\python\envs\jotang-ml
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
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import tempfile as _tempfile
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(_tempfile.gettempdir(), "mplconfig-codex"))

import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
LINE = "=" * 74
DEFAULT_IMAGE = _os.path.join(HERE, "..", "Jotang-ml-task2", "data", "PetImages", "Cat", "0.jpg")


def section(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sdpa(Q, K, V):
    """和 attention_scratch.py 里一模一样的三行：分数 → softmax → 加权平均。"""
    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(Q.shape[-1])
    weights = softmax(scores, axis=-1)
    return scores, weights, weights @ V


def load_image(path, size):
    im = Image.open(path).convert("RGB")
    orig_size = im.size
    im = im.resize((size, size), Image.BICUBIC)
    return np.asarray(im).astype(np.float32) / 255.0, orig_size


def to_patches(img, patch):
    """(H, W, 3) → (N, patch*patch*3)：把图切成不重叠的小块，每块展平成一个向量。"""
    h, w, c = img.shape
    gh, gw = h // patch, w // patch
    p = img.reshape(gh, patch, gw, patch, c).transpose(0, 2, 1, 3, 4)
    return p.reshape(gh * gw, patch * patch * c), gh, gw


def main():
    ap = argparse.ArgumentParser(description="ViT：图像怎么变成 Transformer 的输入")
    ap.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    ap.add_argument("--size", type=int, default=224, help="把图缩放到多少像素（ViT 用 224）")
    ap.add_argument("--patch", type=int, default=16, help="patch 边长（ViT 用 16 或 32）")
    ap.add_argument("--d-model", type=int, default=64, help="演示用的 token 维度（ViT-Base 是 768）")
    ap.add_argument("--patch-row", type=int, default=-1, help="要看哪个 patch 的注意力（-1 = 自动挑）")
    ap.add_argument("--patch-col", type=int, default=-1)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--out", type=str, default="06_vit_patches.png")
    args = ap.parse_args()

    section("拓展 3：ViT —— 把图像当成一句话来读")

    img_path = _os.path.abspath(args.image)
    if not _os.path.exists(img_path):
        raise SystemExit(f"找不到图片：{img_path}（可以用 --image 指定别的图）")
    img, orig_size = load_image(img_path, args.size)
    patches, gh, gw = to_patches(img, args.patch)
    n_tokens = patches.shape[0]
    dim_raw = patches.shape[1]
    print(f"原图：{_os.path.basename(img_path)}  原始尺寸 {orig_size[0]}×{orig_size[1]}")
    print(f"缩放后：{args.size}×{args.size}，patch = {args.patch}×{args.patch}")
    print(f"→ patch 网格 {gh}×{gw} = {n_tokens} 个 patch，每个 patch 展平 = "
          f"{args.patch}·{args.patch}·3 = {dim_raw} 维")

    rng = np.random.default_rng(0)
    # 线性投影：ViT 里这是一个学出来的大矩阵（768 → 768）；这里用随机正交投影降到 d_model，
    # 保证"相似 patch ↔ 相似 token"，方便看注意力。
    proj, _ = np.linalg.qr(rng.normal(size=(dim_raw, args.d_model)))
    tokens = patches @ proj
    # 所有自然图像 patch 的"平均长相"都很像（都带一点同样的背景色），如果不减掉它，
    # 任意两个 patch 的余弦相似度都会接近 1，注意力看上去就会糊成一片。
    # 减掉平均 patch 相当于只比较"这块 patch 偏离平均长相的方向"，纹理差异立刻显出来。
    tokens = tokens - tokens.mean(axis=0, keepdims=True)
    # 让每个 token 的模长 = √d_model：这样 QKᵀ/√d_k 里的分数就等于 √d_k · cos 相似度，
    # 数量级合适、softmax 既不过饱和也不糊成一团。
    tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True) * np.sqrt(args.d_model)
    # ViT 会把 [CLS] 拼在最前面，最终用它的输出做分类；这里也拼上，看到形状就理解结构了。
    cls_token = rng.normal(size=args.d_model)
    cls_token = cls_token / np.linalg.norm(cls_token) * np.sqrt(args.d_model)
    tokens_all = np.vstack([cls_token[None, :], tokens])          # (1 + N, d_model)
    print(f"加一个 [CLS] token → 序列长度 {tokens_all.shape[0]}（ViT-Base："
          f"196 + 1 = 197 个 token，每个 768 维）")

    # patch 之间的注意力：Q = K = V = token 序列（这就是 encoder 的自注意力）
    _, weights, _ = sdpa(tokens_all, tokens_all, tokens_all)
    W = weights[1:, 1:]                                           # 去掉 [CLS] 那一行一列
    print(f"注意力权重矩阵：{weights.shape}（含 CLS），patch 之间是 {W.shape} = "
          f"{n_tokens}×{n_tokens}；ViT 每一层都要算一次这么大的矩阵")

    if args.patch_row >= 0 and args.patch_col >= 0:
        qr, qc = args.patch_row, args.patch_col
    else:                                                         # 自动挑"纹理最丰富"的 patch
        std_map = patches.reshape(gh, gw, -1).std(axis=2)
        qr, qc = np.unravel_index(np.argmax(std_map), std_map.shape)
    qi = qr * gw + qc
    attn_map = W[qi].reshape(gh, gw)
    order = np.argsort(W[qi])[::-1]
    print(f"\n查询 patch：第 {qr} 行第 {qc} 列（展平后是第 {qi} 个 token）")
    print("它最关注的前几个 patch（编号 = 行×%d + 列，权重）：" % gw)
    for k in order[:args.topk]:
        print(f"  patch {k:>3}（第 {k // gw} 行第 {k % gw} 列）: {W[qi, k]:.4f}")
    print(f"自己对自己的权重 = {W[qi, qi]:.4f}（自注意力里'自己'通常也是一个大头）")
    print(f"\n注意力分布并不均匀：最大 {attn_map.max():.4f}，最小 {attn_map.min():.4f}，"
          f"Top-{args.topk} 个 patch 占了 {attn_map.ravel()[order[:args.topk]].sum():.1%}")

    # ---------------- 画图 ----------------
    fig = plt.figure(figsize=(16.4, 10.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.24, wspace=0.24)
    fig.suptitle("ViT：把图像切成 16×16 的「词」，然后用注意力互相看", fontsize=14, y=0.975)

    # (a) 原图 + patch 网格
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    for t in range(gh + 1):
        ax.axhline(t * args.patch - 0.5, color="#ffffff", lw=0.6, alpha=0.8)
        ax.axvline(t * args.patch - 0.5, color="#ffffff", lw=0.6, alpha=0.8)
    ax.add_patch(Rectangle((qc * args.patch - 0.5, qr * args.patch - 0.5), args.patch,
                           args.patch, fill=False, edgecolor="#c0392b", lw=2.2))
    ax.set_title(f"(a) 输入：{args.size}×{args.size} 的图\n切成 {gh}×{gw} 个 "
                 f"{args.patch}×{args.patch} 的 patch（红框 = 下面要看的那一块）",
                 fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    # (b) patch → token 的形状流程
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("(b) 从图像到序列：形状怎么变", fontsize=10.5)
    ax.text(0.02, 0.98,
            f"① 图像        {args.size}×{args.size}×3\n"
            f"     ↓ 切 patch（不重叠、按行扫）\n"
            f"② patch 矩阵  {gh}×{gw} = {n_tokens} 个 × {dim_raw} 维\n"
            f"     ↓ 线性投影（ViT 里是学出来的）\n"
            f"③ token 序列  {n_tokens} × {args.d_model}\n"
            f"     ↓ 前面拼一个 [CLS]\n"
            f"④ 输入序列    {tokens_all.shape[0]} × {args.d_model}\n"
            f"     ↓ + 位置编码（ViT 用可学习的那套）\n"
            f"     ↓ 12 层 encoder（每层：多头自注意力 + FFN）\n"
            f"     ↓ 只取 [CLS] 位置的输出接一层线性分类头\n"
            f"⑤ 输出        类别概率",
            fontsize=9.2, va="top", linespacing=1.75)
    ax.text(0.02, 0.0,
            "关键点：patch 之间没有重叠、没有卷积，每一层里\n"
            "每个 patch 都能直接看到其它所有 patch ——\n"
            "CNN 要叠很多层才能把感受野扩到全图，ViT 第一层\n"
            "就是全局的（代价：计算量 ~ 序列长度的平方）。",
            fontsize=8.8, va="bottom", linespacing=1.55, color="#8e2b1c")

    # (c) 注意力热力图（patch 网格）
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(attn_map, cmap="magma")
    ax.add_patch(Rectangle((qc - 0.5, qr - 0.5), 1, 1, fill=False, edgecolor="#00d0ff", lw=2.2))
    ax.set_title(f"(c) 第 ({qr},{qc}) 个 patch 的注意力分布\n"
                 f"（共 {n_tokens} 个 key，每行权重和为 1）", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # (d) 叠加回原图
    ax = fig.add_subplot(gs[1, 0])
    up = np.kron(attn_map, np.ones((args.patch, args.patch)))
    up = up / up.max()
    ax.imshow(img)
    ax.imshow(up, cmap="magma", alpha=0.55)
    ax.add_patch(Rectangle((qc * args.patch - 0.5, qr * args.patch - 0.5), args.patch,
                           args.patch, fill=False, edgecolor="#00d0ff", lw=2.2))
    ax.set_title("(d) 把注意力叠回原图\n亮的地方 = 这个 patch 正在向谁借信息", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])

    # (e) 被关注最多的 patch 长什么样
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title(f"(e) 权重最高的 {args.topk} 个 patch（query 在最左边）", fontsize=10.5)
    kk = args.topk
    for i, idx in enumerate(order[:kk]):
        pr, pc = idx // gw, idx % gw
        thumb = patches[idx].reshape(args.patch, args.patch, 3)
        ax.imshow(thumb, extent=(i * 1.05, i * 1.05 + 0.95, 0.56, 0.96))
        ax.text(i * 1.05 + 0.47, 0.50, f"{W[qi, idx]:.2f}", ha="center", fontsize=8.2)
        ax.text(i * 1.05 + 0.47, 0.42, f"({pr},{pc})", ha="center", fontsize=7.2,
                color="#555555")
    thumb_q = patches[qi].reshape(args.patch, args.patch, 3)
    ax.imshow(thumb_q, extent=(0.0, 0.95, 0.13, 0.37))
    ax.text(1.05, 0.25, "← query patch 自己（权重最大，因为和自己最像）", fontsize=8.6,
            va="center")
    ax.set_xlim(-0.1, kk * 1.05 + 0.3)
    ax.set_ylim(0.0, 1.02)
    ax.text(0, 0.06,
            "相似度越高、注意力越集中；被关注的这几块也是图上相似的纹理。\n"
            "未训练时这些'被关注的 patch'就是长得像的 patch；\n"
            "训练之后，头会分化成'看同类物体''看边缘''看整体轮廓'等。",
            fontsize=8.4, va="top", color="#333333")

    # (f) 文字：ViT 与原论文的三点不同
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.set_title("(f) ViT 与原论文的三个不同", fontsize=10.5)
    ax.text(0.0, 0.98,
            "① 输入换了：句子 → 图像\n"
            "   词 → patch（16×16，展平成 768 维）\n"
            "   词序 → patch 的二维位置（可学习的位置向量）\n"
            "② 结构删了：ViT 只用 Encoder\n"
            "   原论文是 encoder-decoder（机器翻译）\n"
            "   ViT 不需要 decoder：图像分类是一个\n"
            "   「一个输入 → 一个标签」的任务\n"
            "   分类靠 [CLS] 那一位置的输出接一层线性头\n"
            "③ 训练方式变了：必须大规模预训练\n"
            "   原论文：4.5M 句对、8 张 P100、12 小时\n"
            "   ViT：ImageNet-21k（14M 图）或\n"
            "   JFT-300M（300M 图）预训练，再微调\n"
            "   在小数据集上直接训，ViT 不如 CNN ——\n"
            "   因为 CNN 自带「局部性 + 平移等变」的先验，\n"
            "   ViT 什么先验都没有，只能靠数据学。",
            fontsize=8.9, va="top", linespacing=1.62)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path = _os.path.join(HERE, args.out)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    print("\n一句话总结：ViT 把「一个句子 = 一串词」换成「一张图 = 一串 patch」，")
    print("剩下的注意力机制和原论文一模一样；删掉的只有 decoder。")


if __name__ == "__main__":
    main()
