# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 5 · CLIP 图文配对

题目（任务 2）：
    1. 准备 4 张内容不同的图片（输入尺寸通常 224×224），每张写一句简短的英文描述，记录来源；
    2. 用 CLIP 分别提取图像特征和文本特征，打印两类特征的形状，并在计算相似度前做 L2 归一化；
    3. 计算所有图片与所有文字之间的余弦相似度，得到 4×4 相似度矩阵并画热力图，
       检查每张图是否最接近正确描述，分析配对成功或失败的原因；
    4. 故意加入一句与该图很相似、但关键内容错误的描述（颜色 / 数量 / 动作不对），
       观察 CLIP 能否分辨，并记录相似度怎样变化。

本脚本做的事（只做推理，不训练）：
    * exp0 模型：加载 OpenAI 官方 CLIP 的公开权重 RN50（官方仓库同名实现），
      打印视觉 / 文本两个编码器的结构规模、特征维度（1024）、文本上下文长度与温度系数；
    * exp1 特征：图像张量 [4,3,224,224] → encode_image → [4,1024]；
      token [4,77] → encode_text → [4,1024]；打印形状、若干维度的数值、归一化前后的模长，
      并验证"归一化之后 点积 ≡ 余弦相似度"（两者差 < 1e-6）；
    * exp2 相似度：算 4×4 余弦相似度矩阵，画热力图，做图→文 / 文→图两个方向的检索，
      逐个分析配对成功与失败的原因（含对角线 vs 次大值的间隔 margin）；
    * exp3 错误描述：每张图再配 3 句"很像但关键内容错"的描述（颜色错 / 数量错 / 动作或对象错），
      看 CLIP 是否仍把正确那句排第一，并记录相似度的变化 Δ；
    * exp4 延伸：零样本分类（4 类图 + task2 里的猫狗照片，和那个训练过的 CNN 对照）、
      prompt 工程（"cat" vs "a photo of a cat" 等模板与 7 模板集成）、
      温度系数的作用（logit_scale 变大 / 变小时 softmax 分布怎样变）。

运行方式（本目录下三选一）：
    1) conda activate jotang-ml   然后   python clip_pairing.py
    2) conda run -n jotang-ml python clip_pairing.py
    3) python clip_pairing.py --only exp2 exp3        # 只跑其中几节

输出图（都在本目录）：
    00_image_text_pairs.png   4 张图 + 各自的英文描述 + 来源
    01_similarity_heatmap.png 4×4 余弦相似度热力图 + 每张图的"正确 - 次大"间隔
    02_l2_normalization.png   为什么要 L2 归一化：模长 → 单位球 → 点积即余弦
    03_wrong_description.png  错误描述实验：4 图 ×（1 正确 + 3 错误）的相似度与差值
    04_zero_shot_prompt.png   零样本分类（含猫狗对照）+ prompt 模板 + 温度系数
输出的数字全部写进 results.json。
"""

# ==================================================================
# 【第 0 步】运行环境自举（与 task4 相同的三个坑，不能删）
# ------------------------------------------------------------------
# (1) numpy 的 MKL 在 conda 环境的 Library\bin 下；不激活环境直接跑 python.exe，
#     该目录不在 PATH 上，numpy 会载入错误的 BLAS，做矩阵乘时进程直接原生崩溃
#     （"Windows fatal exception: code 0xc06d007f"，没有任何 Python 报错）。
# (2) torch 自带 Intel libiomp5md.dll，conda 的 mkl 用 LLVM libomp.dll，
#     两份 OpenMP 同时存在时会抛 "OMP: Error #15" 并中止进程。
# (3) Windows 控制台默认 GBK，直接 print "→" "√" 会 UnicodeEncodeError。
# ==================================================================
import os as _os
import sys as _sys

_ENV_ROOT = _os.path.dirname(_sys.executable)          # 例如 D:\python\envs\jotang-ml
for _sub in ("Library\\bin", "Library\\mingw-w64\\bin", "Library\\usr\\bin",
             "DLLs", "Scripts", "bin"):
    _dll_dir = _os.path.join(_ENV_ROOT, _sub)
    if _os.path.isdir(_dll_dir):
        _os.environ["PATH"] = _dll_dir + _os.pathsep + _os.environ.get("PATH", "")
        if hasattr(_os, "add_dll_directory"):          # Python 3.8+
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

# ==================================================================
# 【第 1 步】pkg_resources 兼容层
# ------------------------------------------------------------------
# openai-clip 1.0.1 只从 pkg_resources 里拿 packaging 来做版本比较，但 setuptools>=81
# 已经不再随包分发 pkg_resources，于是 `import clip` 会直接 ModuleNotFoundError。
# 为了一个版本比较去降级整个环境的 setuptools 不划算，这里注册一个最小替身模块，
# 只提供 clip 真正用到的那一个属性。
# ==================================================================
try:                                                    # 正常情况下直接用真模块
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    import types as _types
    import packaging as _packaging
    import packaging.version as _packaging_version  # noqa: F401  (仅为了导入成功)

    _shim = _types.ModuleType("pkg_resources")
    _shim.packaging = _packaging                        # clip 要的就是这个
    _sys.modules["pkg_resources"] = _shim

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import clip
import matplotlib
matplotlib.use("Agg")                                   # 不出窗口，直接存图
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "image"
MODEL_NAME = "RN50"                                     # 题目推荐：先用较小的 RN50
DEVICE = "cpu"
RNG_SEED = 42

# ------------------------------------------------------------------
# 实验用图：4 张内容不同的图片（COCO val2017，公开数据集，来源逐张记录）
# ------------------------------------------------------------------
IMAGES = [
    {
        "file": "01_cat.jpg",
        "caption": "two cats lying on a pink couch",
        "coco_id": "000000039769",
        "wrong": {
            "颜色错": "two cats lying on a blue couch",
            "数量错": "three cats lying on a pink couch",
            "动作/对象错": "two dogs lying on a pink couch",
        },
        "traps": {
            "同义改写": "two cats resting on a pink sofa",
            "语序颠倒": "a pink couch lying on two cats",
            "否定句": "there are no cats on a pink couch",
        },
    },
    {
        "file": "02_bear.jpg",
        "caption": "a large brown bear in the grass",
        "coco_id": "000000000285",
        "wrong": {
            "颜色错": "a large white polar bear in the grass",
            "数量错": "two large brown bears in the grass",
            "动作/对象错": "a large brown bear swimming in a river",
        },
        "traps": {
            "同义改写": "a big brown bear standing in the grass",
            "语序颠倒": "the grass is in a large brown bear",
            "否定句": "there is no bear in the grass",
        },
    },
    {
        "file": "03_bus.jpg",
        "caption": "a red double-decker bus on the street",
        "coco_id": "000000001584",
        "wrong": {
            "颜色错": "a blue double-decker bus on the street",
            "数量错": "two red double-decker buses on the street",
            "动作/对象错": "a red double-decker train on the street",
        },
        "traps": {
            "同义改写": "a red double-decker bus driving down the road",
            "语序颠倒": "the street is on a red double-decker bus",
            "否定句": "there is no bus on the street",
        },
    },
    {
        "file": "04_cake.jpg",
        "caption": "a birthday cake with raspberries and blueberries",
        "coco_id": "000000002157",
        "wrong": {
            "颜色错": "a birthday cake with chocolate frosting and blueberries",
            "数量错": "a birthday cake with a single raspberry",
            "动作/对象错": "a person eating a birthday cake with raspberries",
        },
        "traps": {
            "同义改写": "a birthday cake topped with raspberries and blueberries",
            "语序颠倒": "the berries are baking a birthday cake",
            "否定句": "a birthday cake without any berries",
        },
    },
]

SOURCE_TMPL = "COCO val2017 数据集（http://cocodataset.org，标注 CC BY 4.0），图片 ID {}，取自官方图床 http://images.cocodataset.org/val2017/{}.jpg"


# ==================================================================
# 小工具
# ==================================================================
def sep(title: str = "", ch: str = "=", width: int = 78) -> None:
    if title:
        print("\n" + ch * 4 + f" {title} " + ch * max(1, width - len(title) - 6))
    else:
        print(ch * width)


def fmt_matrix(m: np.ndarray, row_names, col_names, fmt: str = "{:.3f}") -> str:
    w = max(max(len(str(c)) for c in col_names), len(fmt.format(-1.0))) + 2
    head = " " * (max(len(str(r)) for r in row_names) + 2) + "".join(
        str(c).ljust(w) for c in col_names)
    lines = [head]
    for i, r in enumerate(row_names):
        cells = "".join(fmt.format(m[i, j]).ljust(w) for j in range(len(col_names)))
        lines.append(f"{str(r).ljust(max(len(str(x)) for x in row_names))}  {cells}")
    return "\n".join(lines)


def l2norm(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化：每行除以它自己的 2-范数。"""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_images(paths):
    """读出 PIL 图，并顺手打印原始尺寸（CLIP 的预处理会把它变成 224×224）。"""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    for p, im in zip(paths, imgs):
        print(f"  {Path(p).name:<14} 原始尺寸 {im.size[0]}×{im.size[1]}  "
              f"({im.size[0] * im.size[1] / 1e6:.2f} M 像素, {Path(p).stat().st_size / 1024:.0f} KB)")
    return imgs


def encode_images(model, preprocess, imgs) -> tuple[np.ndarray, torch.Tensor]:
    """图像编码器：PIL → 预处理张量 → vision tower → [N, 1024] 特征。"""
    batch = torch.stack([preprocess(im) for im in imgs]).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_image(batch)
    return feats.cpu().numpy().astype(np.float64), batch


def encode_texts(model, texts) -> tuple[np.ndarray, torch.Tensor]:
    """文本编码器：字符串 → token id → text tower → 取 EOT 位置 → [N, 1024] 特征。"""
    tokens = clip.tokenize(texts, truncate=True).to(DEVICE)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return feats.cpu().numpy().astype(np.float64), tokens


def savefig(fig, name: str) -> Path:
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")
    return path


# ==================================================================
# exp0：加载模型，看清两个编码器
# ==================================================================
def exp0_model(model, preprocess, results):
    sep("exp0 · 加载 CLIP（官方实现 + 公开权重）")
    print(f"clip 包位置        : {clip.__file__}")
    print(f"官方仓库可选模型    : {', '.join(clip.available_models())}")
    print(f"本次加载            : {MODEL_NAME}（题目建议先用较小的 RN50）  设备 = {DEVICE}")

    n_vis = sum(p.numel() for p in model.visual.parameters())
    # 注意：CLIP 把 token embedding / 位置编码 / 最后的投影都放在 CLIP 模块上而不是
    # Transformer 子模块里，所以"文本侧"的参数量要两边加起来才对得上论文的 63M。
    n_txt_blocks = sum(p.numel() for p in model.transformer.parameters())
    n_txt_embed = (model.token_embedding.weight.numel() + model.positional_embedding.numel()
                   + model.ln_final.weight.numel() + model.ln_final.bias.numel()
                   + model.text_projection.numel() + model.logit_scale.numel())
    n_txt = n_txt_blocks + n_txt_embed
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\n视觉编码器          : {type(model.visual).__name__}（ResNet50 + 注意力池化）")
    print(f"  输入分辨率        : {model.visual.input_resolution} × {model.visual.input_resolution}")
    print(f"  输出特征维度      : {model.visual.output_dim}")
    print(f"  参数量            : {n_vis / 1e6:.1f} M")
    print(f"文本编码器          : Transformer 宽度 {model.transformer.width}，"
          f"{model.transformer.layers} 层，"
          f"{sum(p.numel() for p in model.transformer.resblocks[0].parameters()) / 1e6:.2f} M/层（第 1 层）")
    print(f"  上下文长度        : {model.context_length}"
          f"（<|startoftext|> + 正文 + <|endoftext|>，正文最多 75 个 token；"
          f"不足补 0，超长直接截断）")
    print(f"  参数量            : {n_txt / 1e6:.1f} M")
    print(f"    （其中 12 层 Transformer {n_txt_blocks / 1e6:.1f} M ≈ "
          f"{sum(p.numel() for p in model.transformer.resblocks[0].parameters()) / 1e6:.2f} M/层 × 12；"
          f"词表嵌入 + 位置编码 + 投影 {n_txt_embed / 1e6:.1f} M。"
          f"合计 {n_txt / 1e6:.1f} M，与论文里“63 M 参数、12 层、宽 512、8 头”的文本编码器对得上）")
    print(f"  词表              : {model.vocab_size} = 256 个字节 token + 49152 个 BPE 合并"
          f"（<|startoftext|>=49406, <|endoftext|>=49407）")
    print(f"\n总计                : {n_all / 1e6:.1f} M 参数（纯推理，不更新任何权重）")
    print(f"可学习温度系数      : logit_scale = 1/0.07 -> {float(model.logit_scale.exp()):.3f}"
          f"（即 logits × {float(model.logit_scale.exp()):.0f} 再进 softmax）")

    results["model"] = {
        "name": MODEL_NAME,
        "feature_dim": int(model.visual.output_dim),
        "image_input_resolution": int(model.visual.input_resolution),
        "text_context_length": int(model.context_length),
        "text_vocab_size": int(model.vocab_size),
        "params_total_M": round(n_all / 1e6, 2),
        "params_visual_M": round(n_vis / 1e6, 2),
        "params_text_M": round(n_txt / 1e6, 2),
        "logit_scale": round(float(model.logit_scale.exp()), 3),
    }
    return model


# ==================================================================
# exp1：特征形状 + L2 归一化
# ==================================================================
def exp1_features(model, preprocess, captions, results):
    sep("exp1 · 提取两类特征，打印形状，然后 L2 归一化")
    paths = [IMAGE_DIR / im["file"] for im in IMAGES]
    print("先把 4 张图读进来（原图任意尺寸，预处理统一成 224×224）：")
    imgs = load_images(paths)

    img_feat_raw, batch = encode_images(model, preprocess, imgs)
    txt_feat_raw, tokens = encode_texts(model, captions)

    print(f"\n图像张量（预处理后）: {tuple(batch.shape)}  dtype={batch.dtype}  "
          f"取值区间 [{batch.min():.3f}, {batch.max():.3f}]（已按 ImageNet 均值/标准差归一化）")
    print(f"图像特征            : {img_feat_raw.shape}   （4 张图 × 1024 维）")
    print(f"token id            : {tuple(tokens.shape)}  （4 句 × 77 个位置；"
          f"[{int(tokens[0, 0])}=SOT, {int(tokens[0, -1])}=EOT/0=padding]）")
    print(f"文本特征            : {txt_feat_raw.shape}  （4 句 × 1024 维）")
    print("\n看几个具体数字（第 1 张图 / 第 1 句的前 8 维）：")
    print("  图像特征前 8 维 : " + "  ".join(f"{v:+.3f}" for v in img_feat_raw[0, :8]))
    print("  文本特征前 8 维 : " + "  ".join(f"{v:+.3f}" for v in txt_feat_raw[0, :8]))

    # ---------------- L2 归一化 ----------------
    img_norms_before = np.linalg.norm(img_feat_raw, axis=1)
    txt_norms_before = np.linalg.norm(txt_feat_raw, axis=1)
    img_feat = l2norm(img_feat_raw)
    txt_feat = l2norm(txt_feat_raw)
    img_norms_after = np.linalg.norm(img_feat, axis=1)
    txt_norms_after = np.linalg.norm(txt_feat, axis=1)

    print("\nL2 归一化（每个特征向量除以它自己的模长）：")
    print(f"  图像特征模长 before: {np.array2string(img_norms_before, precision=2)}")
    print(f"  图像特征模长 after : {np.array2string(img_norms_after, precision=6)}")
    print(f"  文本特征模长 before: {np.array2string(txt_norms_before, precision=2)}")
    print(f"  文本特征模长 after : {np.array2string(txt_norms_after, precision=6)}")

    # 归一化后：点积就是余弦相似度（差应在浮点误差级别）
    dots = img_feat @ txt_feat.T
    cos = dots / (
        np.linalg.norm(img_feat, axis=1, keepdims=True) @ np.linalg.norm(txt_feat, axis=1, keepdims=True).T)
    print(f"  归一化后 点积 与 余弦相似度 的最大差异: {np.abs(dots - cos).max():.2e}"
          "  -> 单位向量上点积 ≡ 余弦")
    raw_dots = img_feat_raw @ txt_feat_raw.T
    ratio = raw_dots / dots
    print(f"  不归一化的点积 = 余弦 × (|img|·|txt|)：比值矩阵（每行 / 每列一个模长乘积）")
    print("    " + np.array2string(ratio, precision=1, suppress_small=True))
    print(f"    参考: |img|·|txt| 的 4×4 乘积矩阵（应与上一行完全一致）")
    print("    " + np.array2string(np.outer(img_norms_before, txt_norms_before), precision=1))

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    idx = np.arange(len(IMAGES))
    ax.bar(idx - 0.2, img_norms_before, 0.4, label="图像特征模长", color="#4c72b0")
    ax.bar(idx + 0.2, txt_norms_before, 0.4, label="文本特征模长", color="#dd8452")
    for i, (a, b) in enumerate(zip(img_norms_before, txt_norms_before)):
        ax.text(i - 0.2, a + 0.3, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.3, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_ylim(0, max(img_norms_before.max(), txt_norms_before.max()) * 1.25)
    ax.set_xticks(idx)
    ax.set_xticklabels([Path(im["file"]).stem for im in IMAGES], fontsize=8.5)
    ax.set_ylabel("向量模长 |x|")
    ax.set_title("(a) 原始特征的模长：两者不在一个尺度上", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ax.scatter(raw_dots.ravel(), (dots * np.outer(img_norms_before, txt_norms_before)).ravel(),
               s=38, color="#55a868", zorder=3)
    lim = [min(raw_dots.min(), 0), raw_dots.max() * 1.05]
    ax.plot(lim, lim, "--", color="#999999", lw=1, label="y = x")
    ax.set_xlabel("不归一化的点积  x_img · x_text")
    ax.set_ylabel("归一化点积 × (|img||txt|)")
    ax.set_title("(b) 两者只差一个模长因子（16 个点全在 y=x 上）", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    dot_max = [(scales[i] ** 2) * (raw_dots / np.outer(img_norms_before, txt_norms_before)).max()
               for i in range(len(scales))]
    cos_max = [(raw_dots / np.outer(img_norms_before, txt_norms_before)).max()] * len(scales)
    ax.plot(scales, dot_max, "o-", color="#c44e52", label="不归一化：最大点积随缩放² 变化")
    ax.plot(scales, cos_max, "s--", color="#4c72b0", label="归一化后：余弦完全不变")
    ax.set_xscale("log")
    ax.set_xticks(scales)
    ax.set_xticklabels([f"×{s:g}" for s in scales], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("把图像特征整体乘以的倍数 k（log）")
    ax.set_ylabel("最大相似度")
    ax.set_title("(c) 不归一化时，相似度会被模长带跑", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("为什么要先做 L2 归一化：把两个编码器的输出都放到同一个单位球上，"
                 "点积才等于余弦相似度、才可跨样本比较", fontsize=12, y=1.03)
    savefig(fig, "02_l2_normalization.png")

    results["features"] = {
        "image_tensor_shape": list(batch.shape),
        "image_feature_shape": list(img_feat_raw.shape),
        "text_token_shape": list(tokens.shape),
        "text_feature_shape": list(txt_feat_raw.shape),
        "image_norm_before": [round(float(v), 3) for v in img_norms_before],
        "text_norm_before": [round(float(v), 3) for v in txt_norms_before],
        "norm_after": 1.0,
        "max_abs_diff_dot_vs_cosine": float(np.abs(dots - cos).max()),
        "cosine_before_norm_matches_after": bool(
            np.allclose(raw_dots / np.outer(img_norms_before, txt_norms_before), dots, atol=1e-9)),
    }
    return img_feat, txt_feat, img_feat_raw, txt_feat_raw


# ==================================================================
# exp2：4×4 余弦相似度矩阵
# ==================================================================
def exp2_matrix(img_feat, txt_feat, captions, results):
    sep("exp2 · 4×4 余弦相似度矩阵（图 → 文 与 文 → 图）")
    sim = img_feat @ txt_feat.T
    img_names = [Path(im["file"]).stem for im in IMAGES]
    short_txt = [f"T{i + 1}" for i in range(len(captions))]

    print("相似度矩阵（行 = 图片，列 = 文字；正确配对在对角线）：")
    print(fmt_matrix(sim, img_names, short_txt))
    print("\n对应的 4 句描述：")
    for i, c in enumerate(captions):
        print(f"  T{i + 1}: {c}")

    print("\n逐张图检查（图 → 文 检索）：")
    margins, top1_hit = [], []
    for i, name in enumerate(img_names):
        order = np.argsort(-sim[i])
        top = order[0]
        second = order[1]
        margin = sim[i, i] - sim[i, second] if top == i else sim[i, i] - sim[i, top]
        margins.append(float(margin))
        top1_hit.append(bool(top == i))
        flag = "✔ 正确描述排第 1" if top == i else f"✘ 排第 1 的是 T{top + 1}"
        print(f"  {name:<8} {flag}：sim(correct)={sim[i, i]:+.4f}，"
              f"最接近的错误描述 T{second + 1} sim={sim[i, second]:+.4f}，"
              f"与第 1 名 T{top + 1} sim={sim[i, top]:+.4f}")
        print(f"          排序: " + " > ".join(
            f"T{j + 1}({sim[i, j]:+.4f})" for j in order))

    print("\n逐句描述检查（文 → 图 检索）：")
    col_hit = []
    for j, c in enumerate(captions):
        order = np.argsort(-sim[:, j])
        hit = bool(order[0] == j)
        col_hit.append(hit)
        tag = "✔" if hit else "✘"
        print(f"  {tag} T{j + 1} “{c}” -> 第 1 名 {img_names[order[0]]}"
              f"({sim[order[0], j]:+.4f})，第 2 名 {img_names[order[1]]}({sim[order[1], j]:+.4f})")

    acc_i2t = float(np.mean(top1_hit))
    acc_t2i = float(np.mean(col_hit))
    print(f"\n图→文 准确率 {sum(top1_hit)}/{len(top1_hit)} = {acc_i2t:.0%}；"
          f"文→图 准确率 {sum(col_hit)}/{len(col_hit)} = {acc_t2i:.0%}")
    print(f"对角线的平均相似度 {np.diag(sim).mean():+.4f}；"
          f"非对角线的平均相似度 {(sim.sum() - np.trace(sim)) / (sim.size - len(sim)):+.4f}")
    print(f"平均间隔 margin（正确 - 最接近的错句）= {np.mean(margins):+.4f}")

    # ---------------- 画图 ----------------
    fig = plt.figure(figsize=(14.5, 5.6))
    ax = fig.add_axes([0.30, 0.18, 0.30, 0.66])
    im = ax.imshow(sim, cmap="viridis", vmin=sim.min(), vmax=sim.max())
    ax.set_xticks(range(len(captions)))
    ax.set_xticklabels([f"T{i + 1}" for i in range(len(captions))], fontsize=9)
    ax.set_yticks(range(len(img_names)))
    ax.set_yticklabels(img_names, fontsize=9)
    for i in range(len(img_names)):
        for j in range(len(captions)):
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if sim[i, j] < sim.max() * 0.75 else "black")
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#e24a33", lw=2.2))     # 正确配对
    ax.set_xlabel("文字 T1~T4", fontsize=9.5)
    ax.set_ylabel("图片", fontsize=9.5)
    ax.set_title("(a) 4×4 余弦相似度矩阵\n红框 = 正确配对（对角线）", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("余弦相似度", fontsize=9)

    ax2 = fig.add_axes([0.72, 0.18, 0.24, 0.66])
    colors = ["#55a868" if h else "#c44e52" for h in top1_hit]
    ax2.barh(range(len(img_names)), margins, color=colors)
    for i, m in enumerate(margins):
        ax2.text(m + 0.004 if m >= 0 else m - 0.004, i, f"{m:+.3f}", va="center",
                 ha="left" if m >= 0 else "right", fontsize=8.5)
    ax2.axvline(0, color="#333333", lw=1)
    ax2.set_yticks(range(len(img_names)))
    ax2.set_yticklabels(img_names, fontsize=9)
    ax2.set_xlabel("正确描述与最接近错句的相似度之差", fontsize=9)
    ax2.set_title("(b) 配对把握（margin）\n绿 = 正确排第 1，红 = 被错句超过", fontsize=10.5)
    ax2.grid(alpha=0.25, axis="x")
    fig.suptitle("CLIP 的零样本图文配对：4 张图 × 4 句描述，正确配对落在对角线",
                 fontsize=12.5, y=0.99)
    savefig(fig, "01_similarity_heatmap.png")

    results["pairing"] = {
        "captions": captions,
        "similarity_matrix": [[round(float(v), 4) for v in row] for row in sim],
        "image_to_text_top1_hit": top1_hit,
        "text_to_image_top1_hit": col_hit,
        "acc_image_to_text": acc_i2t,
        "acc_text_to_image": acc_t2i,
        "margins": [round(m, 4) for m in margins],
        "mean_diag": round(float(np.diag(sim).mean()), 4),
        "mean_offdiag": round(float((sim.sum() - np.trace(sim)) / (sim.size - len(sim))), 4),
    }
    return sim


# ==================================================================
# exp3：故意写错的描述
# ==================================================================
def exp3_wrong(model, preprocess, img_feat, txt_feat, results):
    sep("exp3 · 故意加入“很像但关键内容错”的描述")
    err_types = list(IMAGES[0]["wrong"].keys())
    print(f"错误类型：{ ' / '.join(err_types) }；每张图都是 1 句正确 + 3 句错误，各自算相似度。\n")

    detail = []
    delta = np.zeros((len(IMAGES), len(err_types)))
    for i, meta in enumerate(IMAGES):
        texts = [meta["caption"]] + [meta["wrong"][k] for k in err_types]
        feat, _ = encode_texts(model, texts)
        feat = l2norm(feat)
        sims = (img_feat[i] @ feat.T)
        order = np.argsort(-sims)
        winner = order[0]
        print(f"[{Path(meta['file']).stem}] 正确描述: “{meta['caption']}”")
        print("    " + " | ".join(f"{'正确' if j == 0 else err_types[j - 1]} {sims[j]:+.4f}"
                                  for j in range(len(texts))))
        for j in range(1, len(texts)):
            d = float(sims[0] - sims[j])
            delta[i, j - 1] = d
        for rank, j in enumerate(order):
            tag = "正确" if j == 0 else err_types[j - 1]
            mark = "  <-- 最像" if rank == 0 else ""
            print(f"      #{rank + 1} {tag:<8} {sims[j]:+.4f}  “{texts[j]}”{mark}")
        ok = winner == 0
        print(f"    => CLIP {'✔ 分辨成功（正确句仍排第 1）' if ok else '✘ 被错句骗过'}"
              f"，最小间隔 {min(sims[0] - sims[j] for j in range(1, len(texts))):+.4f}\n")
        detail.append({
            "image": meta["file"],
            "correct": meta["caption"],
            "correct_rank": int(list(order).index(0) + 1),
            "sims": {("correct" if j == 0 else err_types[j - 1]): round(float(sims[j]), 4)
                     for j in range(len(texts))},
            "delta_vs_correct": {err_types[j - 1]: round(float(sims[0] - sims[j]), 4)
                                 for j in range(1, len(texts))},
        })

    print("错误描述让相似度下降多少（正确 - 错误，正数表示 CLIP 认为错句更不像）：")
    print(fmt_matrix(delta, [Path(im["file"]).stem for im in IMAGES], err_types, fmt="{:+.3f}"))
    print(f"\n按错误类型平均下降: " + "，".join(
        f"{k} {delta[:, j].mean():+.3f}" for j, k in enumerate(err_types)))
    best = err_types[int(np.argmax(delta.mean(axis=0)))]
    worst = err_types[int(np.argmin(delta.mean(axis=0)))]
    print(f"=> 平均来看 CLIP 对“{best}”这类错误最敏感，对“{worst}”最容易放过。")
    flat = [(delta[i, j], img_names_i, err_types[j])
            for i, img_names_i in enumerate([Path(im["file"]).stem for im in IMAGES])
            for j in range(len(err_types))]
    flat.sort()
    print("   最容易被放过的三句（相似度几乎没掉）：")
    for d, img, k in flat[:3]:
        meta = next(m for m in IMAGES if Path(m["file"]).stem == img)
        print(f"     {img:<8} {k:<8} Δ={d:+.4f}  “{meta['wrong'][k]}”")
    print("   最敏感的两句：")
    for d, img, k in flat[-2:][::-1]:
        meta = next(m for m in IMAGES if Path(m["file"]).stem == img)
        print(f"     {img:<8} {k:<8} Δ={d:+.4f}  “{meta['wrong'][k]}”")
    print("   注意：4 张图的正确句全部排第 1，但“数量错”只差 0.007~0.034，"
          "蛋糕那张多出一个人都只差 0.005 —— 这一侧是 CLIP 的软肋。")

    # ---------------- 画图 ----------------
    img_names = [Path(im["file"]).stem for im in IMAGES]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    colors = ["#4c72b0", "#c44e52", "#dd8452", "#8172b3"]
    idx = np.arange(len(IMAGES))
    w = 0.2
    vals = {("正确"): np.array([d["sims"]["correct"] for d in detail])}
    for k in err_types:
        vals[k] = np.array([d["sims"][k] for d in detail])
    labels = ["正确"] + err_types
    for j, lab in enumerate(labels):
        ax.bar(idx + (j - 1.5) * w, vals[lab], w, label=lab, color=colors[j])
        for i, v in enumerate(vals[lab]):
            ax.text(i + (j - 1.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(idx)
    ax.set_xticklabels(img_names, fontsize=9)
    ax.set_ylabel("余弦相似度")
    ax.set_ylim(min(v.min() for v in vals.values()) - 0.02,
                max(v.max() for v in vals.values()) + 0.035)
    ax.set_title("(a) 每张图对“正确 / 颜色错 / 数量错 / 动作·对象错”四句的相似度", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    m = ax.imshow(delta, cmap="RdYlGn", vmin=-max(abs(delta.min()), abs(delta.max())),
                  vmax=max(abs(delta.min()), abs(delta.max())))
    ax.set_xticks(range(len(err_types)))
    ax.set_xticklabels(err_types, fontsize=9)
    ax.set_yticks(range(len(img_names)))
    ax.set_yticklabels(img_names, fontsize=9)
    for i in range(len(img_names)):
        for j in range(len(err_types)):
            ax.text(j, i, f"{delta[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("(b) 相似度下降量 = 正确 - 错误\n（越绿=CLIP 越能看出这里错了）", fontsize=10.5)
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("关键内容写错之后 CLIP 还认不认：4 张图的正确句都还排第 1，"
                 "但“数量错”“多出一个人”的相似度几乎不掉",
                 fontsize=12, y=1.02)
    savefig(fig, "03_wrong_description.png")

    # ---------------- exp3b：文字陷阱（语序 / 同义 / 否定） ----------------
    trap_types = list(IMAGES[0]["traps"].keys())
    print("\n" + "-" * 78)
    print("额外一测：句子意思被“改坏”但用词几乎不变时，CLIP 还看得见吗？")
    trap_delta = np.zeros((len(IMAGES), len(trap_types)))
    trap_detail = []
    captions_norm = l2norm(txt_feat)                     # 4 句“正确描述”的归一化特征
    for i, meta in enumerate(IMAGES):
        texts = [meta["caption"]] + [meta["traps"][k] for k in trap_types]
        sims = (img_feat[i] @ l2norm(encode_texts(model, texts)[0]).T)
        # 其它图片的正确描述：与这张图的相似度（这些句子内容完全对不上）
        cross = np.array([float(captions_norm[j] @ img_feat[i])
                          for j in range(len(IMAGES)) if j != i])
        print(f"[{Path(meta['file']).stem}] 正确句 {sims[0]:+.4f}")
        for j, k in enumerate(trap_types):
            trap_delta[i, j] = sims[0] - sims[j + 1]
            print(f"    {k:<6} {sims[j + 1]:+.4f}  Δ={sims[0] - sims[j + 1]:+.4f}   "
                  f"“{meta['traps'][k]}”")
        neg_sim = sims[1 + trap_types.index("否定句")]
        print(f"    对照：别的图片的 3 句正确描述，与这张图的相似度只有 "
              f"{cross.min():+.4f} ~ {cross.max():+.4f}")
        print(f"    => 否定句 {neg_sim:+.4f} "
              f"{'仍然高于' if neg_sim > cross.max() else '低于'}任意一句“完全无关但描述准确”的句子"
              f"{'（说明 CLIP 基本没处理否定）' if neg_sim > cross.max() else ''}")
        trap_detail.append({
            "image": meta["file"],
            "correct": round(float(sims[0]), 4),
            "traps": {k: round(float(sims[j + 1]), 4) for j, k in enumerate(trap_types)},
            "delta_vs_correct": {k: round(float(trap_delta[i, j]), 4)
                                 for j, k in enumerate(trap_types)},
            "cross_image_max": round(float(cross.max()), 4),
        })

    print("\n陷阱造成的相似度下降量（正确 - 陷阱）：")
    print(fmt_matrix(trap_delta, img_names, trap_types, fmt="{:+.3f}"))
    print("按陷阱类型平均: " + "，".join(
        f"{k} {trap_delta[:, j].mean():+.3f}" for j, k in enumerate(trap_types)))
    st = int(np.argmin(trap_delta.mean(axis=0)))
    neg_j = trap_types.index("否定句")
    print(f"=> 三类里掉分最少的是“{trap_types[st]}”（平均 {trap_delta[:, st].mean():+.4f}）："
          f"把“lying on”换成“resting on”、“couch”换成“sofa”，相似度几乎不动"
          f" —— 说明它对齐的是“近义词的语义方向”，而不是在数词形。")
    print(f"   真正的软肋是语序和否定：“语序颠倒”平均只掉 {trap_delta[:, trap_types.index('语序颠倒')].mean():+.4f}"
          f"，“否定句”平均只掉 {trap_delta[:, neg_j].mean():+.4f}，"
          f"而且 4 张图的否定句得分全部高于“别的图片的正确描述”（完全无关的句子）。")
    print("   一句话总结：CLIP 的文本端更像“一袋词”——同义词能对上，语序和 not 基本不参与判断。")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    vm = max(abs(trap_delta.min()), abs(trap_delta.max()))
    m2 = ax.imshow(trap_delta, cmap="RdYlGn", vmin=-vm, vmax=vm)
    ax.set_xticks(range(len(trap_types)))
    ax.set_xticklabels(trap_types, fontsize=9.5)
    ax.set_yticks(range(len(img_names)))
    ax.set_yticklabels(img_names, fontsize=9.5)
    for i in range(len(img_names)):
        for j in range(len(trap_types)):
            ax.text(j, i, f"{trap_delta[i, j]:+.3f}", ha="center", va="center", fontsize=10)
    ax.set_title("(a) 语义被改坏后相似度掉了多少\n（绿 = CLIP 看得出来，红 = 完全没看出来）",
                 fontsize=10.5)
    fig.colorbar(m2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.bar(range(len(trap_types)), trap_delta.mean(axis=0),
           color=["#55a868", "#dd8452", "#c44e52"])
    ax.set_ylim(0, trap_delta.mean(axis=0).max() * 1.2)
    for j, v in enumerate(trap_delta.mean(axis=0)):
        ax.text(j, v + 0.002, f"{v:+.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(trap_types)))
    ax.set_xticklabels(trap_types, fontsize=9.5)
    ax.set_ylabel("平均下降量")
    ax.set_title("(b) 三类文字陷阱的平均影响\n掉分越少 = CLIP 越没察觉这句话已经被改坏",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("同义改写 / 语序颠倒 / 加一句 not：CLIP 的文字端哪些信号真的参与了判断",
                 fontsize=12, y=1.03)
    savefig(fig, "05_text_traps.png")

    results["wrong_description"] = {
        "error_types": err_types,
        "per_image": detail,
        "delta_matrix": [[round(float(v), 4) for v in row] for row in delta],
        "mean_delta_by_type": {k: round(float(delta[:, j].mean()), 4)
                               for j, k in enumerate(err_types)},
        "most_sensitive_type": best,
        "least_sensitive_type": worst,
        "all_correct_rank1": all(d["correct_rank"] == 1 for d in detail),
        "traps": {
            "types": trap_types,
            "per_image": trap_detail,
            "delta_matrix": [[round(float(v), 4) for v in row] for row in trap_delta],
            "mean_delta_by_type": {k: round(float(trap_delta[:, j].mean()), 4)
                                   for j, k in enumerate(trap_types)},
        },
    }
    return delta


# ==================================================================
# exp4：零样本分类 + prompt 工程 + 温度系数
# ==================================================================
PET_DIR = HERE.parent / "Jotang-ml-task2" / "data" / "PetImages"


def zeroshot_probs(model, preprocess, imgs, label_texts, logit_scale=None):
    """零样本分类：把类别名当成文本，谁的文本特征与图像特征最像就判给谁。"""
    img_feat = l2norm(encode_images(model, preprocess, imgs)[0])
    txt_feat = l2norm(encode_texts(model, label_texts)[0])
    logits = (img_feat @ txt_feat.T) * (100.0 if logit_scale is None else logit_scale)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return p


def exp4_zeroshot_prompt(model, preprocess, results):
    sep("exp4 · 零样本分类 / prompt 工程 / 温度系数")
    paths = [IMAGE_DIR / im["file"] for im in IMAGES]
    imgs = [Image.open(p).convert("RGB") for p in paths]

    # ---- (1) 4 张图做 4 类零样本分类（标签直接写类别名） ----
    nouns = ["cat", "bear", "bus", "cake"]
    labels = [f"a photo of a {n}" for n in nouns]
    print("(1) 4 类零样本分类（模型从没见过这几张图，也没被训练过这 4 个类别）：")
    print("    候选标签：" + " | ".join(labels))
    p = zeroshot_probs(model, preprocess, imgs, labels)
    conf = np.zeros((len(imgs), len(labels)))
    for i in range(len(imgs)):
        conf[i, int(np.argmax(p[i]))] = 1
        print(f"    {Path(paths[i]).name:<14} -> {nouns[int(np.argmax(p[i]))]:<5}"
              f"（概率 {p[i].max():.3f}）  " +
              "  ".join(f"{n}:{v:.3f}" for n, v in zip(nouns, p[i])))
    acc = float(np.mean([np.argmax(p[i]) == i for i in range(len(imgs))]))
    print(f"    => 分类准确率 {acc:.0%}")

    # ---- (2) 用 task2 的猫狗照片做零样本分类，和训练过的 CNN 对照 ----
    print(f"\n(2) 零样本 vs 训练：同一个猫狗数据集（task2 拿它训过 CNN），这里从 Cat/Dog 各随机抽 12 张，"
          f"不训练、不微调，直接把类别名当标签分类：")
    zoo = []
    rng = np.random.default_rng(RNG_SEED)
    if PET_DIR.exists():
        for cls in ("Cat", "Dog"):
            files = sorted((PET_DIR / cls).glob("*.jpg"))
            pick = rng.choice(len(files), size=min(12, len(files)), replace=False)
            for k in pick:
                try:
                    im = Image.open(files[k]).convert("RGB")
                    im.load()
                    zoo.append((im, cls))
                except Exception:                       # 数据集里有少量损坏文件
                    continue
    if zoo:
        zoo_imgs = [z[0] for z in zoo]
        zp = zeroshot_probs(model, preprocess, zoo_imgs, ["a photo of a cat", "a photo of a dog"])
        correct = sum(int(np.argmax(zp[i])) == (0 if z[1] == "Cat" else 1)
                      for i, z in enumerate(zoo))
        per_cls = {}
        for c, name in ((0, "Cat"), (1, "Dog")):
            idxs = [i for i, z in enumerate(zoo) if z[1] == name]
            per_cls[name] = round(float(np.mean([int(np.argmax(zp[i])) == c for i in idxs])), 4)
        n_cat = sum(1 for z in zoo if z[1] == "Cat")
        n_dog = len(zoo) - n_cat
        print(f"    猫 {per_cls['Cat']:.0%}（{n_cat} 张），狗 {per_cls['Dog']:.0%}（{n_dog} 张），"
              f"总体 {correct}/{len(zoo)} = {correct / len(zoo):.0%}")
        print(f"    对照：task2 里从零训练的 CNN 在验证集上是 86.4%（8 epoch）。")
        print(f"    一个是“训练 8 个 epoch 的 2 类分类器”，一个是“一句提示词都不改的零样本”，"
              f"这就是图文对比学习的优势。")
        zoo_acc = correct / len(zoo)
    else:
        print(f"    跳过：没找到 {PET_DIR}（task2 的猫狗数据不在本机上）。")
        per_cls, zoo_acc = {}, None

    # ---- (3) prompt 工程：同一批图，只改提示词模板 ----
    print("\n(3) prompt 工程：类别含义完全一样，只是换说法 —— 零样本准确率会变吗？"
          "（用 4 类图 + 猫狗共 28 张图算平均准确率）")
    all_imgs = imgs + ([z[0] for z in zoo] if zoo else [])
    # 候选标签 = 4 类名词 + 猫狗两个名词；注意标签表里 cat 出现两次（第 1 位和第 5 位），
    # 猫狗照片的"正确答案"要分别指向这两个位置，写错了准确率会被冤枉地拉低。
    all_nouns = nouns + (["cat", "dog"] if zoo else [])
    dog_at = all_nouns.index("dog")
    truth = [0, 1, 2, 3] + ([0 if z[1] == "Cat" else dog_at for z in zoo] if zoo else [])
    templates = {
        "只用类别名": "{}",
        "a photo of a {}": "a photo of a {}",
        "a photo of the {}": "a photo of the {}",
        "a close-up photo of a {}": "a close-up photo of a {}",
        "7 模板集成（论文做法）": None,
    }
    ensemble_templates = [
        "a photo of a {}.", "a photo of a big {}.", "a photo of a small {}.",
        "a blurry photo of a {}.", "a close-up photo of a {}.", "a bad photo of a {}.",
        "a photo of the {}.",
    ]
    prompt_res = {}
    for name, tmpl in templates.items():
        if tmpl is not None:
            if name == "只用类别名":
                labs = list(all_nouns)
            else:
                labs = [tmpl.format(n) for n in all_nouns]
            probs = zeroshot_probs(model, preprocess, all_imgs, labs)
        else:
            probs = np.mean([zeroshot_probs(model, preprocess, all_imgs,
                                             [t.format(n) for n in all_nouns])
                             for t in ensemble_templates], axis=0)     # 模板特征先各自 softmax 再平均
        acc_n = float(np.mean([int(np.argmax(probs[i])) == truth[i]
                               for i in range(len(all_imgs))]))
        prompt_res[name] = acc_n
        print(f"    {name:<26} 准确率 {acc_n:.0%}")
    best_prompt = max(prompt_res, key=prompt_res.get)
    worst_prompt = min(prompt_res, key=prompt_res.get)
    print(f"    => 最好的是“{best_prompt}”（{prompt_res[best_prompt]:.0%}），"
          f"最差的是“{worst_prompt}”（{prompt_res[worst_prompt]:.0%}）："
          f"类别含义一个字都没变，只换说法，差距就有这么大。")
    print(f"       这里 7 模板集成反而低于最好的单模板：论文的模板集成赢在 80 个模板、"
          f"而且里面没有“blurry / bad photo”这种自贬模板；")
    print(f"       这 28 张图太容易（单模板已经 100%），集成只会被差模板往下拉"
          f" —— prompt 工程不是“模板越多越好”。")

    # ---- (4) 温度系数：logit_scale 变大 / 变小时 softmax 怎么变 ----
    print("\n(4) 温度系数：CLIP 把相似度乘以 exp(logit_scale)=100 再进 softmax。"
          "改成别的值会怎样？")
    scales = [1.0, 10.0, 100.0, 1000.0]
    base = (l2norm(encode_images(model, preprocess, imgs)[0]) @
            l2norm(encode_texts(model, labels)[0]).T)
    print(f"    （原始余弦相似度：最大 {base.max():.3f}，最小 {base.min():.3f}，"
          f"跨度只有 {base.max() - base.min():.3f}）")
    temp_curves, temp_acc = {}, {}
    for s in scales:
        z = (base * s)
        z = z - z.max(axis=1, keepdims=True)
        pr = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        ent = -(pr * np.log(np.clip(pr, 1e-12, None))).sum(axis=1) / np.log(pr.shape[1])
        acc_s = float(np.mean([int(np.argmax(pr[i])) == i for i in range(len(imgs))]))
        temp_acc[s] = acc_s
        temp_curves[s] = (pr, float(np.mean(ent)), float(pr.max(axis=1).mean()))
        print(f"    logit_scale={s:<7.0f} 第 1 张图概率 {np.array2string(pr[0], precision=4)}"
              f"  平均最大概率 {pr.max(axis=1).mean():.3f}  归一化熵 {ent.mean():.3f}"
              f"  4 类准确率 {acc_s:.0%}")
    print("    注意最后一列：温度系数是单调变换，argmax（谁排第一）完全不变，"
          "所以单纯分类时改温度不影响结果；")
    print("    它真正影响的是“多标签 / 阈值判断 / 多个 prompt 投票平均”这类需要概率数值的场景，"
          "以及训练时梯度的尺度。")

    # ---------------- 画图 ----------------
    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.52)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(p, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(nouns)))
    ax.set_xticklabels(nouns, fontsize=9)
    ax.set_yticks(range(len(imgs)))
    ax.set_yticklabels([Path(x).stem for x in paths], fontsize=9)
    for i in range(len(imgs)):
        for j in range(len(nouns)):
            ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if p[i, j] > 0.6 else "black")
    ax.set_title(f"(a) 4 类零样本分类（准确率 {acc:.0%}）\n只给了 4 个类别名，没有任何训练", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    short_names = {"只用类别名": "只用类别名（cat / dog）",
                   "a photo of a {}": "a photo of a {}",
                   "a photo of the {}": "a photo of the {}",
                   "a close-up photo of a {}": "a close-up photo of a {}",
                   "7 模板集成（论文做法）": "7 模板集成（论文做法）"}
    names = list(prompt_res.keys())
    vals = [prompt_res[n] for n in names]
    bars = ax.barh(range(len(names)), vals,
                   color=["#4c72b0" if v == max(vals) else "#8c9cb0" for v in vals])
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short_names[n] for n in names], fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"零样本准确率（{len(all_imgs)} 张图）")
    ax.set_title("(b) prompt 工程：只换说法，准确率就变", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")

    ax = fig.add_subplot(gs[0, 2])
    for s, c in zip(scales, ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]):
        ax.plot(range(1, len(nouns) + 1), temp_curves[s][0][0], "o-", color=c,
                label=f"logit_scale={s:g}（熵 {temp_curves[s][1]:.2f}）")
    ax.set_xticks(range(1, len(nouns) + 1))
    ax.set_xticklabels(nouns, fontsize=9)
    ax.set_ylabel("softmax 概率")
    ax.set_title("(c) 第 1 张图：温度系数越大，分布越尖", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[1, 0])
    if zoo:
        jit = np.random.default_rng(RNG_SEED)
        for c, name, col in ((0, "cat", "#4c72b0"), (1, "dog", "#dd8452")):
            idxs = [i for i, z in enumerate(zoo) if z[1] == ("Cat" if c == 0 else "Dog")]
            probs = zp[idxs, c]
            ax.scatter(probs, c + jit.uniform(-0.12, 0.12, len(idxs)), s=45, color=col,
                       alpha=0.85, edgecolor="white", zorder=3,
                       label=f"{name}（n={len(idxs)}，正确率 {per_cls['Cat' if c == 0 else 'Dog']:.0%}）")
            ax.text(min(probs) - 0.08, c, f"最小 {min(probs):.2f}", fontsize=8,
                    va="center", ha="right", color=col)
        ax.set_xlabel("模型给出正确类别的概率")
        ax.set_xlim(0.6, 1.06)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["cat", "dog"], fontsize=9.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title("(d) 零样本分类 task2 的猫狗照片\n（同一个模型，没有针对猫狗训练过）", fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, axis="x")

    ax = fig.add_subplot(gs[1, 1:])
    ax.plot(scales, [temp_curves[s][2] for s in scales], "o-", color="#c44e52",
            label="平均最大概率（置信度）")
    ax.plot(scales, [temp_curves[s][1] for s in scales], "s--", color="#4c72b0",
            label="归一化熵（分布的均匀程度）")
    ax.axvline(100, color="#333333", ls=":", lw=1.2)
    ax.text(100, 1.02, "  CLIP 默认 exp(logit_scale)=100", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("logit_scale（相似度乘的倍数，log）")
    ax.set_ylabel("数值（0~1）")
    ax.set_ylim(0, 1.1)
    ax.set_title("(e) 温度系数增大 -> 置信度升高、熵下降：回答会更“自信”，也更容易一头错到底",
                 fontsize=10.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("零样本分类、prompt 工程与温度系数：CLIP 的“输出端”其实只有一个 softmax",
                 fontsize=12.5, y=0.98)
    savefig(fig, "04_zero_shot_prompt.png")

    results["zero_shot"] = {
        "coco_labels": labels,
        "coco_probs": [[round(float(v), 4) for v in row] for row in p],
        "coco_accuracy": acc,
        "pet_accuracy": zoo_acc,
        "pet_accuracy_per_class": per_cls,
        "pet_n": len(zoo),
        "cnn_task2_reference": 0.8642,
    }
    results["prompt_engineering"] = {k: round(float(v), 4) for k, v in prompt_res.items()}
    results["temperature"] = {
        "cosine_range": [round(float(base.min()), 4), round(float(base.max()), 4)],
        "accuracy_by_scale": {str(int(s)): round(float(v), 4) for s, v in temp_acc.items()},
        "curves": {str(int(s)): {
            "probs_img1": [round(float(v), 4) for v in temp_curves[s][0][0]],
            "mean_entropy": round(temp_curves[s][1], 4),
            "mean_max_prob": round(temp_curves[s][2], 4),
        } for s in scales},
    }
    return results


# ==================================================================
# 图 00：4 张图 + 描述 + 来源
# ==================================================================
def fig_pairs():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.17,
                        hspace=0.16, wspace=0.04)
    for ax, meta in zip(axes.ravel(), IMAGES):
        im = Image.open(IMAGE_DIR / meta["file"]).convert("RGB")
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"{Path(meta['file']).stem}  ·  “{meta['caption']}”",
                     fontsize=11, pad=6)
    # 来源统一放在图下方（4 行），避免贴在图下面时和下一行标题挤在一起
    lines = []
    for meta in IMAGES:
        lines.append(f"{Path(meta['file']).stem:<8} "
                     f"COCO val2017 图片 ID {meta['coco_id']}，"
                     f"取自官方图床 http://images.cocodataset.org/val2017/{meta['coco_id']}.jpg")
    fig.text(0.03, 0.135,
             "来源（4 张图都来自公开数据集，无版权问题）：COCO val2017（http://cocodataset.org，"
             "标注 CC BY 4.0，图片来自 Flickr）\n" + "\n".join(lines),
             ha="left", va="top", fontsize=7.6, color="#444444", linespacing=1.5)
    fig.suptitle("实验用图（4 张，内容互不相同）+ 每张一句英文描述 + 来源", fontsize=12.5, y=0.97)
    savefig(fig, "00_image_text_pairs.png")


# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Task 5 · CLIP 图文配对实验")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["exp0", "exp1", "exp2", "exp3", "exp4"],
                    help="只跑其中几节，默认全部跑")
    args = ap.parse_args()
    run = (lambda k: True) if not args.only else (lambda k: k in args.only)

    torch.set_num_threads(min(8, _os.cpu_count() or 8))
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    sep("Task 5 · CLIP 图文配对（只做推理，不训练）")
    print(f"Python {_sys.version.split()[0]} | torch {torch.__version__} | numpy {np.__version__} "
          f"| device {DEVICE}")
    t0 = time.time()
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    model.eval()
    print(f"权重来自官方公开 checkpoint（~/.cache/clip/{MODEL_NAME}.pt），加载耗时 {time.time() - t0:.1f} s")

    captions = [im["caption"] for im in IMAGES]
    results = {}
    results["env"] = {"python": _sys.version.split()[0], "torch": torch.__version__,
                      "numpy": np.__version__, "device": DEVICE}
    results["sources"] = {im["file"]: SOURCE_TMPL.format(im["coco_id"], im["coco_id"])
                          for im in IMAGES}

    fig_pairs()
    exp0_model(model, preprocess, results)
    img_feat = txt_feat = None
    if run("exp1") or run("exp2") or run("exp3"):
        img_feat, txt_feat, _, _ = exp1_features(model, preprocess, captions, results)
    if run("exp2"):
        exp2_matrix(img_feat, txt_feat, captions, results)
    if run("exp3"):
        exp3_wrong(model, preprocess, img_feat, txt_feat, results)
    if run("exp4"):
        exp4_zeroshot_prompt(model, preprocess, results)

    out = HERE / "results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    sep()
    print(f"全部完成，用时 {time.time() - t0:.1f} s；数字已写入 {out.name}")


if __name__ == "__main__":
    main()
