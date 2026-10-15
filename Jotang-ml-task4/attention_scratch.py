# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 4 · 手写注意力

题目（拓展 1 + 拓展 2）：
    1. 用 NumPy 或 PyTorch 手写一个最小的**单头** Scaled Dot-Product Attention，
       输出 Attention 分数矩阵和权重矩阵，并画成热力图（输入用随机数即可）。
       主体计算不许直接调用 nn.MultiheadAttention。
    2. 在 1 的基础上拓展为**多头**注意力。

本脚本做的事：
    * 只用一个 @ 号写清楚 Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V，并打印每一步的全部数字；
    * 用 6 个中文词当 token、d_k = 4 的小例子，把分数矩阵、除以 √d_k 的分数矩阵、
      softmax 权重矩阵、输出矩阵逐个打印出来（数字小到能肉眼验算）；
    * 验算：每行权重之和 = 1、权重全为正、输出 = 权重的加权平均；
    * 把同一套权重搬进 PyTorch 重算一遍，逐项对比 NumPy 与 torch 的结果（最大绝对误差）；
    * 再用 torch.autograd 拿 d(loss)/d(W_q)，与中心差分数值梯度对拍（说明整条注意力链可微）；
    * 可选：和 nn.MultiheadAttention 对拍一次——**只作为第三方参照**，主体实现不使用它；
    * 用 6 个 d_k（4 → 1024）做实验，量化"为什么不缩放会出事"：QKᵀ 的标准差随 d_k 增长，
      未缩放时 softmax 被推进饱和区（最大权重→1、熵→0、梯度→0），除以 √d_k 后恢复正常；
    * 多头：同一输入、h = 4 个头，(a) 随机初始化权重看 4 个头； (b) 人为构造 4 个有语义的头
      （看自己 / 看前一个词 / 均匀看全部 / 看重复出现的同一个词），说明"多头 ≠ 重复做同一件事"；
    * 位置编码：画出论文的正弦编码矩阵；用"相邻位移的点积恒定"验证它的相对位置性质；
      再用一个小实验证明没有位置编码时注意力对词序完全无感（打乱词序 → 输出只是跟着换位置）；
    * Masked Attention：同一个权重矩阵，做 mask 前后对比，量出"能看到未来"的泄漏比例。

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python attention_scratch.py
    2) conda run -n jotang-ml python attention_scratch.py
    3) python attention_scratch.py --h 4 --limit 0.5      # limit = 无 mask 时泄漏概率的阈值示例

输出图：
    01_sdpa_heatmap.png     单头：分数矩阵 → 缩放 → softmax 权重 → 加权求和
    02_scale_effect.png     为什么必须除以 √d_k
    03_multihead.png        多头：随机初始化的 4 个头 + 人为构造的 4 个有语义的头
    04_pos_encoding.png     位置编码：正弦矩阵 + 相对位移 + "没有 PE 就看不见词序"
    05_causal_mask.png      Masked Attention：mask 前后的权重与泄漏量
"""

# ==================================================================
# 【第 0 步】运行环境自举（关键，不要删）
# ------------------------------------------------------------------
# 现象：不激活 conda 环境而直接运行 python.exe 时，进程做矩阵乘（np.dot / @）
#       的一瞬间会原生崩溃，报 "Windows fatal exception: code 0xc06d007f"，
#       没有任何 Python 报错。
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
        if hasattr(_os, "add_dll_directory"):          # Python 3.8+
            try:
                _os.add_dll_directory(_dll_dir)
            except OSError:
                pass

# 本环境里同时装着两个 OpenMP 运行时：conda-forge 的 mkl 用 LLVM 的 libomp.dll，
# 而 torch 自带 Intel 的 libiomp5md.dll。torch 先加载 libiomp5 之后，numpy 第一次
# 真正调用 BLAS（matplotlib 存图内部也会用到）时 MKL 再去初始化 libomp.dll，
# 就会直接抛 "OMP: Error #15" 并中止进程，而且没有任何 Python 报错。
# 这是 Intel/LLVM 官方给出的开关，只影响“允许多份 OpenMP 共存”，不改变数值结果。
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 控制台编码兜底：Windows 的 cmd/PowerShell 默认是 GBK(936)，直接 print "√" "ᵀ" "→"
# 这类字符会抛 UnicodeEncodeError。把 stdout/stderr 强制成 UTF-8 就与终端无关了。
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# matplotlib 第一次画图要在用户目录下建字体缓存；在受限环境下可能没权限而报警告。
# 指到系统临时目录里，警告消失，也不影响出图。
import tempfile as _tempfile
_os.environ.setdefault("MPLCONFIGDIR", _os.path.join(_tempfile.gettempdir(), "mplconfig-codex"))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                                   # 不出窗口，直接存图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
LINE = "=" * 74


def section(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def show(name, M, fmt="{:5.2f}"):
    """把矩阵打成能肉眼读的一小块文本。"""
    M = np.asarray(M, dtype=float)
    print(f"{name}  shape={M.shape}")
    for row in M:
        print("  [" + " ".join(fmt.format(v) for v in row) + "]")


# ==================================================================
# 一、核心公式：三段代码，就是论文第 3.2 节的全部内容
# ------------------------------------------------------------------
#   scores  = Q Kᵀ / √d_k        分数矩阵：第 i 行是"第 i 个 query 对每个 key 的匹配分"
#   weights = softmax(scores)      权重矩阵：每行归一化成概率分布（和为 1、全为正）
#   output  = weights V            输出：把 V 按权重加权平均
# 公式来自论文式 (1)：Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
# ==================================================================
def softmax(x, axis=-1):
    """数值稳定的 softmax：先减去最大值，指数不会溢出（结果完全等价）。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sdpa_scores(Q, K):
    """分数矩阵 = QKᵀ/√d_k。d_k 是 query/key 的维度，也就是"每个词被打分的向量长度"。"""
    d_k = Q.shape[-1]
    return (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_k)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """单头 Scaled Dot-Product Attention（NumPy 手写版，没有任何库函数帮忙）。"""
    scores = sdpa_scores(Q, K)
    if mask is not None:
        # mask 是布尔矩阵，True = 允许看（保留），False = 禁止看（设成 -∞）
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return scores, weights, weights @ V


def split_heads(X, n_heads):
    """(n, d_model) → (h, n, d_k)：把 d_model 维的向量切成 h 段，一段给一个头。"""
    n, d_model = X.shape
    d_k = d_model // n_heads
    assert d_k * n_heads == d_model, "d_model 必须能被头数整除"
    return X.reshape(n, n_heads, d_k).transpose(1, 0, 2)


def merge_heads(H):
    """(h, n, d_k) → (n, h·d_k)：把头拼回来，恢复成 d_model 维。"""
    n_heads, n, d_k = H.shape
    return H.transpose(1, 0, 2).reshape(n, n_heads * d_k)


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, mask=None, qkv=None):
    """
    多头注意力（NumPy 手写版）。

    与论文式 (2)(3) 一一对应：
        head_i = Attention(X W_i^Q, X W_i^K, X W_i^V)
        MHA(X) = Concat(head_1, ..., head_h) W^O
    注意：这里的"多头"不是把同一个注意力跑 h 次，而是先把向量投影到 h 个不同的子空间
    （每头只有 d_k = d_model/h 维），各头在自己的子空间里算注意力，最后拼接再由 W^O 融合。

    qkv 参数：直接给定 (Q, K, V) 时就不再乘投影矩阵——后面构造"有语义的头"时用得上。
    """
    if qkv is None:
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
    else:
        Q, K, V = qkv
    qh, kh, vh = split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads)

    scores_heads, weight_heads, out_heads = [], [], []
    for i in range(n_heads):
        s, a, o = scaled_dot_product_attention(qh[i], kh[i], vh[i], mask=mask)
        scores_heads.append(s), weight_heads.append(a), out_heads.append(o)

    concat = merge_heads(np.stack(out_heads, axis=0))   # (n, d_model)
    return {
        "Q": Q, "K": K, "V": V,
        "scores": np.stack(scores_heads),               # (h, n, n)
        "weights": np.stack(weight_heads),              # (h, n, n)
        "head_out": np.stack(out_heads),                # (h, n, d_k)
        "concat": concat,                               # (n, d_model)
        "out": concat @ Wo,                             # (n, d_model)
    }


# ==================================================================
# 二、画图小工具
# ==================================================================
def heatmap(ax, M, row_labels, col_labels, title, fmt="{:.2f}", cmap="viridis",
            annot=True, annot_fs=7.2, vmin=None, vmax=None):
    M = np.asarray(M, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("key / value（被看的词）", fontsize=8.5)
    ax.set_ylabel("query（正在看的词）", fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                # 按"这个颜色到底有多亮"决定字色，深色底用白字，浅色底用黑字。
                # （不能用数值大小猜：RdBu 这类发散色图中间是白的，白字会看不见。）
                r, g, b, _ = im.cmap(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=annot_fs,
                        color="white" if lum < 0.55 else "#101010")
    return im


# ==================================================================
# 三、实验 1：单头注意力，把每个矩阵都摊开看
# ==================================================================
def exp1_single_head(seed=7, annotate_numbers=True):
    section("实验 1：手写单头 Scaled Dot-Product Attention（结果全部打印出来）")

    tokens = ["小明", "把", "书", "放在", "桌子", "上"]
    n, d_k = len(tokens), 4
    rng = np.random.default_rng(seed)
    # 随机数就是随机数：注意力机制本身跟输入内容无关，这里只是把"机器怎么算"跑通。
    # 为了让打印出来的数字能读，统一四舍五入到 0.1。
    Q = np.round(rng.normal(0, 1, (n, d_k)), 1)
    K = np.round(rng.normal(0, 1, (n, d_k)), 1)
    V = np.round(rng.normal(0, 1, (n, d_k)), 1)

    print(f"token 序列：{tokens}     d_k = {d_k}     √d_k = {np.sqrt(d_k):.3f}")
    show("Q（query，正在看的词）", Q)
    show("K（key，被看的词）", K)
    show("V（value，被看时提供的值）", V)

    raw = Q @ K.T
    scores, weights, out = scaled_dot_product_attention(Q, K, V)
    show("QKᵀ（未缩放分数）", raw)
    show("QKᵀ/√d_k（论文用的分数）", scores)
    show("softmax(QKᵀ/√d_k)（权重矩阵 A）", weights)
    show("A·V（输出）", out)

    row_sum = weights.sum(axis=1)
    print("\n验算：")
    print(f"  每行权重之和 = {np.round(row_sum, 6)}   （应当全是 1.0）")
    print(f"  权重最小值 = {weights.min():.4f}（应当 > 0，因为指数函数恒正）")
    print(f"  权重的含义：第 i 行的第 j 个数 = 第 i 个词在写自己的表示时，向第 j 个词"
          f"借了多大比例的信息")
    print(f"  输出的含义：O[{tokens.index('放在')}] = Σ_j A[3,j]·V[j]，"
          f"就是 V 按这一行权重的加权平均")

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.4))
    fig.suptitle("单头 Scaled Dot-Product Attention：分数 → 缩放 → softmax → 加权求和",
                 fontsize=13, y=0.98)

    lim = max(abs(raw).max(), abs(scores).max())
    heatmap(axes[0, 0], raw, tokens, tokens, "(a) 分数矩阵 QK^T（未缩放）\n对角线/离群值就是大分",
            cmap="RdBu_r", vmin=-lim, vmax=lim, annot=annotate_numbers)
    heatmap(axes[0, 1], scores, tokens, tokens,
            f"(b) 缩放后 QK^T/√d_k（除以 {np.sqrt(d_k):.2f}）\n数值被压小，softmax 才有梯度",
            cmap="RdBu_r", vmin=-lim, vmax=lim, annot=annotate_numbers)
    # 右边多拼一列"每行之Σ"：softmax 的输出一定归一化，这一列就是它最重要的性质。
    weights_with_sum = np.hstack([weights, row_sum[:, None]])
    cols_sum = tokens + ["Σ 行和"]
    im = heatmap(axes[1, 0], weights_with_sum, tokens, cols_sum,
                 "(c) 权重矩阵 A = softmax(分数)\n最后一列 = 每行之和，恰好全是 1",
                 cmap="viridis", vmin=0, vmax=1, annot=annotate_numbers)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.02)

    ax = axes[1, 1]
    row = tokens.index("放在")
    ax.bar(range(n), weights[row], color="#4a7ea8")
    ax.set_xticks(range(n))
    ax.set_xticklabels(tokens, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("权重", fontsize=9)
    ax.set_title(f"(d) 权重的一行到底在说什么\n查询词「{tokens[row]}」写自己的表示时，"
                 f"从每个词那里取多少", fontsize=10.5)
    for j in range(n):
        ax.text(j, weights[row, j] + 0.02, f"{weights[row, j]:.2f}", ha="center", fontsize=8.5)
    ax.text(0.02, 0.96,
            "输出 O[%d] = Σ_j A[%d,j]·V[j]\n= 按上面这根柱子混合所有词的 value\n"
            "（没有训练过的权重没有语义，这里只看机制）" % (row, row),
            transform=ax.transAxes, va="top", fontsize=8.6, color="#333333")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = _os.path.join(HERE, "01_sdpa_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    return dict(tokens=tokens, Q=Q, K=K, V=V, raw=raw, scores=scores, weights=weights, out=out)


# ==================================================================
# 四、实验 2：为什么必须除以 √d_k
# ------------------------------------------------------------------
# 论文正文与脚注 4 的说法：假设 q、k 的每一维独立、均值 0 方差 1，则 q·k 的方差是 d_k，
# 所以 d_k 一大，点积的绝对值就大，softmax 被推进"几乎 one-hot"的区域，
# 那里的梯度极小，训练会走不动。下面用数字把这句话验一遍。
# ==================================================================
def exp2_scale_effect(seed=0, repeats=64):
    section("实验 2：softmax 的饱和 —— 为什么分数要除以 √d_k")

    d_ks = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    std_raw = np.zeros_like(d_ks, dtype=float)
    max_w_raw = np.zeros_like(d_ks, dtype=float)
    max_w_scaled = np.zeros_like(d_ks, dtype=float)
    entropy_raw = np.zeros_like(d_ks, dtype=float)
    entropy_scaled = np.zeros_like(d_ks, dtype=float)
    n = 32                                                 # 32 个 token 互相看

    rng = np.random.default_rng(seed)
    for idx, d_k in enumerate(d_ks):
        raw_list, max_raw, max_scaled, ent_raw, ent_scaled = [], [], [], [], []
        for _ in range(repeats):
            Q = rng.normal(0, 1, (n, d_k))
            K = rng.normal(0, 1, (n, d_k))
            raw = Q @ K.T
            raw_list.append(raw.std())
            for M, acc_max, acc_ent in ((raw, max_raw, ent_raw),
                                        (raw / np.sqrt(d_k), max_scaled, ent_scaled)):
                W = softmax(M, axis=-1)
                acc_max.append(W.max(axis=1).mean())
                p = W / W.sum(axis=1, keepdims=True)
                acc_ent.append(-(p * np.log(p + 1e-12)).sum(axis=1).mean() / np.log(n))
        std_raw[idx] = np.mean(raw_list)
        max_w_raw[idx] = np.mean(max_raw)
        max_w_scaled[idx] = np.mean(max_scaled)
        entropy_raw[idx] = np.mean(ent_raw)
        entropy_scaled[idx] = np.mean(ent_scaled)

    print(f"{'d_k':>6} {'std(QK^T)':>10} {'√d_k':>8} {'max权重(不缩放)':>16} "
          f"{'max权重(缩放)':>14} {'归一化熵(不缩放)':>18} {'归一化熵(缩放)':>16}")
    for i, d_k in enumerate(d_ks):
        print(f"{d_k:>6} {std_raw[i]:>10.2f} {np.sqrt(d_k):>8.2f} {max_w_raw[i]:>16.4f} "
              f"{max_w_scaled[i]:>14.4f} {entropy_raw[i]:>18.4f} {entropy_scaled[i]:>16.4f}")
    print("\n读法：")
    print("  std(QK^T) ≈ √d_k —— 点积的波动随维度按 √d_k 增长，这就是论文脚注 4 的结论；")
    print("  不缩放：d_k 越大 max 权重越接近 1、归一化熵越接近 0（分布塌成一个尖峰），")
    print("          softmax 落到饱和区 → 梯度≈0 → 参数几乎收不到梯度；")
    print("  缩放后：max 权重和熵基本不随 d_k 变化，注意力保持'有选择但不极端'。")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    fig.suptitle("除以 √d_k 在修什么：让 softmax 不随 d_k 一起塌成尖峰", fontsize=13, y=1.0)

    ax = axes[0]
    ax.plot(d_ks, std_raw, "o-", color="#c0392b", label="实测 std(QK^T)")
    ax.plot(d_ks, np.sqrt(d_ks), "--", color="#555555", label=r"$\sqrt{d_k}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("d_k（log）", fontsize=9.5)
    ax.set_ylabel("标准差（log）", fontsize=9.5)
    ax.set_title("(a) 点积的波动随 d_k 按 √d_k 增长", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(d_ks, max_w_raw, "o-", color="#c0392b", label="不缩放")
    ax.plot(d_ks, max_w_scaled, "s-", color="#2f6f9f", label="除以 √d_k")
    ax.axhline(1.0 / n, color="#888888", ls=":", lw=1)
    ax.text(d_ks[0], 1.0 / n + 0.01, "完全均匀 = 1/n", fontsize=8, color="#666666")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k（log）", fontsize=9.5)
    ax.set_ylabel("每行最大权重（32 个 token 的平均）", fontsize=9.5)
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) 不缩放时 d_k 一大，权重就变成独热向量", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(d_ks, entropy_raw, "o-", color="#c0392b", label="不缩放")
    ax.plot(d_ks, entropy_scaled, "s-", color="#2f6f9f", label="除以 √d_k")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k（log）", fontsize=9.5)
    ax.set_ylabel("归一化熵（1 = 均匀，0 = 独热）", fontsize=9.5)
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) 归一化熵：注意力分布的'均匀程度'", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.06,
            "softmax 越接近独热，\n∂权重/∂分数 越接近 0\n→ 反向传不回去",
            transform=ax.transAxes, fontsize=8.4, color="#8e2b1c")

    fig.tight_layout()
    path = _os.path.join(HERE, "02_scale_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    return dict(d_ks=d_ks, std_raw=std_raw, max_w_raw=max_w_raw, max_w_scaled=max_w_scaled)


# ==================================================================
# 五、实验 3：多头注意力
# ------------------------------------------------------------------
# 论文式 (2)：MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W^O，head_i = Attention(QW_i^Q, …)
# 关键是"h 个不同的投影矩阵"：每个头把 d_model 维的词向量压到 d_k = d_model/h 维的
# 子空间里，因此它能看到的关系类型（"谁指代谁"、"谁修饰谁"）可以和别的头不一样。
# ==================================================================
def build_semantic_heads(tokens, d_model=32, n_heads=4):
    """
    人为构造 4 个"有明确语义"的头，用来解释多头能看到不同关系。

    做法：把词向量设成 8 维的 one-hot 编码，再直接指定 Q、K，使分数矩阵出现固定图案。
    （真实模型里的这些图案是训练出来的，论文附录 Figure 3/4 就是真实模型上学到的例子。）
    """
    n = len(tokens)
    onehot = np.eye(n)                                   # 6 个/8 个词各有一个编号

    def pad(A):                                          # 补零到 d_k 维度
        out = np.zeros((A.shape[0], n))
        out[:, :A.shape[1]] = A
        return out

    # 词的"身份码"：同一个词（在句子里出现两次）共用同一个码，不同词码不同
    ident = np.zeros((n, 6))
    codes = {"喜欢": 0, "猫": 1, "我": 2, "他": 3, "也": 4, "，": 5}
    for i, tok in enumerate(tokens):
        ident[i, codes[tok]] = 1.0

    # 把身份码放大 3 倍：分数 = 9/√d_k，softmax 之后目标位置的权重约 0.7~0.8，
    # 图案才看得清楚（模长小的话 softmax 会把所有位置压成差不多）。
    mag = 3.0
    Q_self, Q_prev = mag * onehot, mag * onehot
    Q_uniform, Q_repeat = np.zeros((n, n)), mag * ident
    K_self = mag * onehot                                # <e_i, e_j> 只在 i = j 时非零
    # <e_i, e_{j+1}> = 1 ⟺ j = i-1（第一个词没有"前一个词"，那一行留全 0 → 权重均匀）
    K_prev = np.zeros((n, n))
    K_prev[:n - 1] = mag * onehot[1:]
    K_uniform = np.zeros((n, n))                         # 所有分数都是 0 → 权重全相等
    K_repeat = mag * ident                               # 同样的身份码 → 互相打高分

    heads_q = [pad(Q_self), pad(Q_prev), pad(Q_uniform), pad(Q_repeat)]
    heads_k = [pad(K_self), pad(K_prev), pad(K_uniform), pad(K_repeat)]
    head_names = ["头① 看自己\n(自己对自己的分最高)", "头② 看前一个词\n(接上文语境)",
                  "头③ 均匀看全部\n(分数全 0 → 1/n)", "头④ 看重复出现的词\n(同一身份码)"]
    return heads_q[:n_heads], heads_k[:n_heads], head_names[:n_heads], d_model


def exp3_multi_head(seed=3, n_heads=4):
    section("实验 3：多头注意力（每个头看自己的子空间）")

    tokens = ["我", "喜欢", "猫", "，", "他", "也", "喜欢", "猫"]
    n, d_model = len(tokens), 32
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d_model))                    # 8 个词的向量表示
    Wq = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wk = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wv = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wo = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))

    res = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads)
    print(f"输入 X: {X.shape} → 线性投影 Q/K/V: {res['Q'].shape} (d_model={d_model})")
    print(f"切成 {n_heads} 个头 → 每个头的 Q: {split_heads(res['Q'], n_heads).shape}"
          f"  (h={n_heads}, n={n}, d_k={d_model // n_heads})")
    print(f"每个头的权重矩阵: {res['weights'].shape}；拼回去 concat: {res['concat'].shape}；"
          f"最后乘 W^O 输出: {res['out'].shape}")
    print(f"参数量口径：多头 = h 个 (d_model×d_k) 的 W_i^Q，合计 h·d_model·d_k "
          f"= d_model²（和单头 d_k=d_model 的一次投影一样多）；多的只是 W^O")
    print("\n各头权重的'尖锐程度'（最大权重，越大越挑）：")
    for i in range(n_heads):
        print(f"  头{i + 1}: max={res['weights'][i].max():.3f}, "
              f"mean={res['weights'][i].mean():.3f}")

    # 人为构造的语义头
    heads_q, heads_k, head_names, _ = build_semantic_heads(tokens, d_model, n_heads)
    V = X
    sem_weights = []
    for q, k in zip(heads_q, heads_k):
        _, w, _ = scaled_dot_product_attention(q, k, V)
        sem_weights.append(w)
    sem_weights = np.stack(sem_weights)
    print("\n人为构造的 4 个头，看两行权重（词「我」= 只出现一次，词「喜欢」= 出现两次）：")
    for row_i in (0, 1):
        print(f"  ── query = 「{tokens[row_i]}」")
        for i, name in enumerate(head_names):
            print(f"     {name.splitlines()[0]}: " +
                  " ".join(f"{t}={v:.2f}" for t, v in zip(tokens, sem_weights[i][row_i])))

    fig, axes = plt.subplots(2, n_heads, figsize=(4.0 * n_heads, 10.2),
                             gridspec_kw={"hspace": 0.46, "wspace": 0.30})
    fig.suptitle("多头注意力：h 个头各有各的投影矩阵，各看各的关系", fontsize=13, y=0.985)
    for i in range(n_heads):
        heatmap(axes[0, i], res["weights"][i], tokens, tokens,
                f"(a{i + 1}) 随机初始化的头 {i + 1}\n未训练 → 图案只是随机的",
                cmap="viridis", vmin=0, vmax=1, annot=False)
    for i in range(n_heads):
        heatmap(axes[1, i], sem_weights[i], tokens, tokens, "(b%d) %s" % (i + 1, head_names[i]),
                cmap="viridis", vmin=0, vmax=1, annot=True, annot_fs=7.8)
    fig.text(0.5, 0.025,
             "行 = query（谁在看），列 = key（被谁看）。上一行：随机权重；"
             "下一行：人为指定的 Q/K，用来展示'不同头可以负责不同关系'。\n"
             "论文表 3 行(A)：单头 24.9 BLEU，8 头 25.8 BLEU —— 一个头做不到，但头太多"
             "（16、32）也掉分，因为每头维度 d_k = d_model/h 被压得太小。",
             ha="center", fontsize=9.2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.925, bottom=0.10)
    path = _os.path.join(HERE, "03_multihead.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    return dict(tokens=tokens, res=res, sem_weights=sem_weights)


# ==================================================================
# 六、实验 4：位置编码
# ------------------------------------------------------------------
# 论文 3.5 节：注意力本身对"词的顺序"完全无感（把输入换顺序，输出只是跟着换顺序），
# 所以要把位置信息加进词向量。论文用固定正弦函数：
#     PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# 作者选它的理由是：任意固定位移 k，PE(pos+k) 都能写成 PE(pos) 的线性函数，
# 于是模型容易学"按相对位置来注意"。
# ==================================================================
def sinusoidal_pe(n_pos, d_model):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe


def exp4_positional_encoding(seed=1, n_pos=60, d_model=64):
    section("实验 4：位置编码 —— 没有它，注意力看不见词的先后")

    pe = sinusoidal_pe(n_pos, d_model)
    print(f"位置编码矩阵 PE: shape={pe.shape}（{n_pos} 个位置 × {d_model} 维）")

    # (1) 相对位移性质：对固定的位移 k，PE(pos)·PE(pos+k) 应当与 pos 无关
    ks = np.arange(1, 13)
    sim_mean, sim_std = [], []
    for k in ks:
        d = np.sum(pe[:n_pos - k] * pe[k:], axis=1)
        sim_mean.append(d.mean())
        sim_std.append(d.std())
    sim_mean, sim_std = np.array(sim_mean), np.array(sim_std)
    print("\n固定位移 k 时，PE(pos)·PE(pos+k) 的均值 / 标准差（标准差≈0 说明只取决于 k）：")
    for k, m, s in list(zip(ks, sim_mean, sim_std))[:6]:
        print(f"  k={k:2d}: 均值={m:7.3f}  标准差={s:.3e}")
    print(f"  …最大标准差 = {sim_std.max():.3e}")

    # (2) 没有 PE 时，打乱词序 = 什么都没变
    rng = np.random.default_rng(seed)
    n, d = 6, 16
    X = rng.normal(0, 1, (n, d))
    Wq = rng.normal(0, 1 / np.sqrt(d), (d, d))
    Wk = rng.normal(0, 1 / np.sqrt(d), (d, d))
    Wv = rng.normal(0, 1 / np.sqrt(d), (d, d))
    perm = np.array([3, 0, 5, 1, 4, 2])                   # 一个打乱

    def one_layer(Xin):
        _, _, out = scaled_dot_product_attention(Xin @ Wq, Xin @ Wk, Xin @ Wv)
        return out

    # 注意实验的公平性：这里"打乱词序"指"词换了位置、位置标签不动"，
    # 而不是"连位置编码也一起搬走"——后者等价于整行一起置换，无论有没有 PE 都一样。
    PE = pe[:n, :d]
    out_plain = one_layer(X)                               # 不加位置编码：原顺序
    out_perm = one_layer(X[perm])                          # 不加位置编码：打乱
    diff_no_pe = np.abs(out_perm - out_plain[perm]).max()  # 把顺序还原后再比

    out_pe = one_layer(X + PE)                             # 加位置编码：原顺序
    out_pe_perm = one_layer(X[perm] + PE)                  # 加位置编码：词换了位置
    diff_pe = np.abs(out_pe_perm - out_pe).max()

    print("\n打乱词序后，位置上的输出变了多少（位置标签不动，只是词换了位置）：")
    print(f"  不加位置编码: 最大差值 = {diff_no_pe:.3e}   → 只是跟着换了位置，"
          f"模型自己完全不知道顺序")
    print(f"  加了位置编码: 最大差值 = {diff_pe:.3e}   → 同样的词放在不同位置，表示不同了")
    print("  这不是因为正弦编码有什么魔法，而是因为它给每个位置打上了不同的标签；")
    print("  换成可学习的位置向量（论文表 3 行 (E)：PPL 4.92 / BLEU 25.7，与正弦几乎一样）"
          "也能达到同样效果。")

    # ---------------- 画图 ----------------
    fig = plt.figure(figsize=(15.2, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])
    fig.suptitle("位置编码（论文 3.5 节）：把'第几个词'变成向量", fontsize=13, y=0.99)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pe[:40], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("维度 i（0…63）", fontsize=9.5)
    ax.set_ylabel("位置 pos（0…39）", fontsize=9.5)
    ax.set_title("(a) 正弦位置编码矩阵\n左侧频率高、右侧频率低（波长 2π → 10000·2π）", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(ks, sim_mean, yerr=sim_std, fmt="o-", color="#2f6f9f", capsize=3,
                label="PE(pos)·PE(pos+k)（对 pos 求均值±标准差）")
    ax.set_xlabel("位移 k（相隔几个词）", fontsize=9.5)
    ax.set_ylabel("相似度（点积）", fontsize=9.5)
    ax.set_title("(b) 相隔 k 个词时，相似度只由 k 决定\n（标准差≈0 → 相对位置是有结构的）", fontsize=10.5)
    ax.legend(fontsize=8.4)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(["不加位置编码", "加了位置编码"], [diff_no_pe, diff_pe],
           color=["#c0392b", "#2f6f9f"])
    for x, v in enumerate([diff_no_pe, diff_pe]):
        ax.text(x, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    top = max(diff_no_pe, diff_pe)
    ax.set_ylim(0, top * 1.25)
    ax.set_ylabel("打乱词序后输出的最大变化", fontsize=9.5)
    ax.set_title("(c) 打乱词序，模型的表示会变吗？\n没 PE 时：完全不变（顺序信息为 0）", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.03, 0.72,
            "注意力是'集合运算'：\n把 6 个词换顺序，输出只是\n跟着换顺序，数值一模一样。\n"
            "要它区分'猫追老鼠'和'老鼠追猫'，\n就必须把位置信息塞进去。",
            transform=ax.transAxes, fontsize=8.6, color="#333333", va="top")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = _os.path.join(HERE, "04_pos_encoding.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    return dict(pe=pe, sim_mean=sim_mean, sim_std=sim_std,
                diff_no_pe=diff_no_pe, diff_pe=diff_pe)


# ==================================================================
# 七、实验 5：Decoder 里的 Masked Attention
# ------------------------------------------------------------------
# 论文 3.2.3 节第三个要点：decoder 的自注意力要阻止"往右看"，
# 做法是在 softmax 之前把非法位置设成 -∞，softmax 之后它们的权重就是 0。
# 为什么必须做？两个层面的原因：
#   1) 自回归的语义：预测第 t 个词时，第 t+1 个词还不存在，训练时它只是"老师给的答案"；
#   2) 训练目标会崩：如果允许看未来，模型只要把答案原样搬过来 loss 就能接近 0，
#      学不到任何"从上下文推测下一个词"的能力，推理时没有答案可抄，效果暴跌。
# ==================================================================
TRAIN_LOG_EVERY = 10


def train_next_token_demo(use_mask, vocab=8, length=8, batch=8,
                          steps=500, lr=0.05, seed=0):
    """
    极小的"预测下一个词"实验，专门用来验证 mask 的必要性。

    token 每一步都重新随机采样：数据里**没有任何规律**，所以"预测下一个词"在信息论上
    就是不可做的，任何正经模型的最好成绩就是均匀猜测 ln(vocab)。
    但如果允许注意力看未来的位置，模型可以学到一条作弊路径：
    "第 i 个位置直接去看第 i+1 个位置的 value，把它抄成答案" —— loss 于是能掉到 0。

    返回：每 TRAIN_LOG_EVERY 步记录一次的 loss 列表。
    """
    d = vocab
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    pos = torch.nn.Parameter(torch.randn(length, d, generator=g) * 0.1)
    Wq = torch.nn.Parameter(torch.randn(d, d, generator=g) / np.sqrt(d))
    Wk = torch.nn.Parameter(torch.randn(d, d, generator=g) / np.sqrt(d))
    Wv = torch.nn.Parameter(torch.randn(d, d, generator=g) / np.sqrt(d))
    Wo = torch.nn.Parameter(torch.randn(d, d, generator=g) / np.sqrt(d))
    Wout = torch.nn.Parameter(torch.randn(d, d, generator=g) / np.sqrt(d))
    params = [pos, Wq, Wk, Wv, Wo, Wout]
    opt = torch.optim.Adam(params, lr=lr)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))

    history = []
    for step in range(steps):
        tok = torch.randint(0, vocab, (batch, length), generator=g)
        X = torch.nn.functional.one_hot(tok, vocab).float() + pos    # 词向量 + 位置向量
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        scores = (Q @ K.transpose(-1, -2)) / np.sqrt(d)
        if use_mask:
            scores = scores.masked_fill(~causal, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ V
        logits = out @ Wout
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), tok[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % TRAIN_LOG_EVERY == 0:
            history.append(float(loss))
    return history


def exp5_causal_mask(seed=11, n=6):
    section("实验 5：Masked Attention —— 为什么不能提前看到后面的答案")

    tokens = ["<s>", "猫", "追", "老鼠", "。" , "猫"] if n == 6 else [f"t{i}" for i in range(n)]
    tokens = tokens[:n]
    d_k = 4
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, (n, d_k))
    K = rng.normal(0, 1, (n, d_k))
    V = rng.normal(0, 1, (n, d_k))

    _, w_open, _ = scaled_dot_product_attention(Q, K, V)            # 没有 mask
    causal = np.tril(np.ones((n, n), dtype=bool))                    # 下三角 = 允许看
    _, w_masked, _ = scaled_dot_product_attention(Q, K, V, mask=causal)

    leak = np.array([w_open[i, i + 1:].sum() for i in range(n)])
    print(f"token：{tokens}")
    show("未加 mask 的权重（每一行和仍为 1，但能看到右边）", w_open)
    show("加了 mask 的权重（右上三角 = 0）", w_masked)
    print("\n每个位置'偷看未来'的概率质量（未加 mask 时）：")
    for i, p in enumerate(leak):
        print(f"  位置 {i}（{tokens[i]}）: 未来占比 = {p:.3f}")
    print(f"  平均泄漏比例 = {leak.mean():.3f}（未训练的权重近似均匀，未来位置天然分到"
          f"约一半；重点不是这个数，而是加 mask 之后它必须是 0）")
    print(f"\n加 mask 后：右上三角最大权重 = {w_masked[np.triu_indices(n, 1)].max():.3e}，"
          f"每行和 = {np.round(w_masked.sum(axis=1), 6)}")
    print("注意第 0 行：只有一个合法位置，softmax 在只有一个非 -∞ 输入时输出恒为 1，")
    print("      所以第一个词只能看自己 —— 这就是'自回归'最严格的一步。")

    # ---- 训练层面的证据：token 完全随机，任何"真本事"都学不到，
    #      但允许看未来的模型能把 loss 抄到 0 ----
    losses_open = train_next_token_demo(use_mask=False, seed=seed)
    losses_masked = train_next_token_demo(use_mask=True, seed=seed)
    print("\n小实验：随机 token 的'预测下一个词'任务（8 个词纯随机，理论上无从预测）")
    print(f"  不加 mask：loss 从 {losses_open[0]:.3f} 掉到 {losses_open[-1]:.4f}"
          f"（≈ 0，因为它直接去看答案）")
    print(f"  加了 mask：loss 从 {losses_masked[0]:.3f} 停在 {losses_masked[-1]:.4f}"
          f"（≈ ln 8 = {np.log(8):.3f}，什么都没学到，这才是诚实的数字）")
    print("  推理时'答案'根本不存在（要预测的词还没生成），所以训练时看到的未来信息")
    print("  在推理时必然消失 —— 这中间的落差就是必须 mask 的根本原因。")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0))
    fig.suptitle("Masked Attention：把'未来'的分数设成 -∞", fontsize=13, y=0.99)

    heatmap(axes[0], w_open, tokens, tokens,
            "(a) 不加 mask：右上三角（红框 = 未来）也分到了权重\n"
            "随机权重下平均 %.0f%% 的概率落在未来" % (100 * leak.mean()),
            cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(i + 1, n):
            axes[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="#c0392b", lw=1.6))

    heatmap(axes[1], w_masked, tokens, tokens,
            "(b) 加了 mask：右上三角 = 0\n每行仍然和为 1（重新分配到可见位置）",
            cmap="viridis", vmin=0, vmax=1)

    ax = axes[2]
    ax.plot(np.arange(len(losses_open)) * TRAIN_LOG_EVERY, losses_open, color="#c0392b",
            label="不加 mask（可以看未来）")
    ax.plot(np.arange(len(losses_masked)) * TRAIN_LOG_EVERY, losses_masked, color="#2f6f9f",
            label="加 causal mask")
    ax.axhline(np.log(8), color="#888888", ls=":", lw=1)
    ax.text(0, np.log(8) + 0.06, "随机猜测的下界 ln 8 = 2.08", fontsize=8.2, color="#666666")
    ax.set_xlabel("训练步数", fontsize=9.5)
    ax.set_ylabel("下一词预测的交叉熵", fontsize=9.5)
    ax.set_ylim(0, max(losses_open[0], losses_masked[0]) * 1.05)
    ax.set_title("(c) 把'预测下一个词'真跑一遍\n8 个词纯随机 → 没有 mask 时 loss 能抄到 0",
                 fontsize=10.5)
    ax.legend(fontsize=8.6)
    ax.grid(alpha=0.25)
    ax.text(0.35, 0.55,
            "没有 mask：模型学会'去看下一个位置'\n（这条'作弊路径'在训练集上完美）\n"
            "有 mask：这条路径被 -∞ 堵死，\nloss 只能停在'瞎猜'的水平",
            transform=ax.transAxes, fontsize=8.4, color="#333333", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = _os.path.join(HERE, "05_causal_mask.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n已保存 {path}")
    return dict(w_open=w_open, w_masked=w_masked, leak=leak)


# ==================================================================
# 八、交叉验证：NumPy 手写 vs PyTorch（含 autograd 与中心差分）
# ==================================================================
def exp6_torch_check(seed=5, n_heads=4, n=6, d_model=32, check_nn_mha=True):
    section("实验 6：交叉验证 —— NumPy 手写 vs PyTorch vs 数值梯度")

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d_model))
    Wq = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wk = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wv = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
    Wo = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))

    np_res = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads)

    # ---- PyTorch 版：同一套公式、同一套权重，逐步对应 ----
    tX = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    tWq = torch.tensor(Wq, dtype=torch.float64, requires_grad=True)
    tWk = torch.tensor(Wk, dtype=torch.float64, requires_grad=True)
    tWv = torch.tensor(Wv, dtype=torch.float64, requires_grad=True)
    tWo = torch.tensor(Wo, dtype=torch.float64, requires_grad=True)

    def torch_mha(X_, Wq_, Wk_, Wv_, Wo_, h):
        n_, d_ = X_.shape
        d_k_ = d_ // h
        Q_, K_, V_ = X_ @ Wq_, X_ @ Wk_, X_ @ Wv_
        # (n, d_model) → (h, n, d_k)：view 成 (n, h, d_k) 再换轴
        qh = Q_.view(n_, h, d_k_).transpose(0, 1)
        kh = K_.view(n_, h, d_k_).transpose(0, 1)
        vh = V_.view(n_, h, d_k_).transpose(0, 1)
        scores_ = (qh @ kh.transpose(-1, -2)) / (d_k_ ** 0.5)
        W_ = torch.softmax(scores_, dim=-1)
        out_ = W_ @ vh
        concat_ = out_.transpose(0, 1).reshape(n_, d_)
        return W_, concat_ @ Wo_

    tW, tOut = torch_mha(tX, tWq, tWk, tWv, tWo, n_heads)
    err_out = float(np.abs(tOut.detach().numpy() - np_res["out"]).max())
    err_w = float(np.abs(tW.detach().numpy() - np_res["weights"]).max())
    print(f"PyTorch 版 vs NumPy 版：输出最大绝对误差 = {err_out:.3e}，"
          f"权重最大绝对误差 = {err_w:.3e}  → 两套实现一致")

    # ---- autograd 梯度 vs 中心差分（只抽查几个元素，够说明"这条链是可微的"）----
    loss = (tOut ** 2).sum()
    grad = torch.autograd.grad(loss, tWq)[0].numpy()

    def loss_of(Wq_np):
        r = multi_head_attention(X, Wq_np, Wk, Wv, Wo, n_heads)
        return float((r["out"] ** 2).sum())

    eps = 1e-6
    picks = [(0, 0), (3, 7), (d_model - 1, d_model - 2)]
    print("\nd(loss)/d(W_q) 抽查（autograd 解析值 vs 中心差分）：")
    worst = 0.0
    for (i, j) in picks:
        Wp = Wq.copy(); Wp[i, j] += eps
        Wm = Wq.copy(); Wm[i, j] -= eps
        num = (loss_of(Wp) - loss_of(Wm)) / (2 * eps)
        worst = max(worst, abs(num - grad[i, j]) / max(1.0, abs(num)))
        print(f"  W_q[{i},{j}]: autograd = {grad[i, j]:+.6f}   中心差分 = {num:+.6f}")
    print(f"  最大相对偏差 = {worst:.2e}  → 注意力整条链（分数 → softmax → 加权 → 投影）都可微")

    # ---- 可选：和 nn.MultiheadAttention 对拍（只作参照，主体实现不使用它）----
    if check_nn_mha:
        try:
            ref = torch.nn.MultiheadAttention(d_model, n_heads, bias=False,
                                              batch_first=False, dtype=torch.float64)
            with torch.no_grad():
                # nn.MultiheadAttention 把 W_q/W_k/W_v 竖着拼成一个 in_proj_weight：
                #   in_proj_weight = [W_q; W_k; W_v]，且 Linear 内部算的是 x·W^T，
                #   而我们的 NumPy 约定是 x·W，所以这里要写 .T（否则对不上）。
                ref.in_proj_weight.copy_(torch.tensor(
                    np.concatenate([Wq.T, Wk.T, Wv.T], axis=0), dtype=torch.float64))
                ref.out_proj.weight.copy_(torch.tensor(Wo.T, dtype=torch.float64))
                seq = torch.tensor(X, dtype=torch.float64).unsqueeze(1)   # (n, batch=1, d)
                ref_out, ref_w = ref(seq, seq, seq, need_weights=True,
                                     average_attn_weights=False)
            d_ref = float(np.abs(ref_out.detach().squeeze(1).numpy() - np_res["out"]).max())
            # need_weights=True, average_attn_weights=False → (batch, heads, n, n)
            d_w = float(np.abs(ref_w.detach().numpy()[0] - np_res["weights"]).max())
            print(f"\nnn.MultiheadAttention（仅作第三方参照）：输出最大误差 = {d_ref:.3e}，"
                  f"权重最大误差 = {d_w:.3e}")
            print("  → 说明手写版和官方实现算的是同一个东西；论文里'多头 = 切子空间 + 拼接 + W^O'"
                  "这句话在代码里就是这样落地的。")
        except Exception as exc:                                     # pragma: no cover
            print(f"\n（跳过 nn.MultiheadAttention 对拍：{exc}）")


# ==================================================================
# 九、主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Task 4：手写 Scaled Dot-Product / Multi-Head Attention")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--h", type=int, default=4, help="多头注意力的头数")
    ap.add_argument("--no-nn-check", action="store_true", help="不跑 nn.MultiheadAttention 对拍")
    ap.add_argument("--only", type=str, default="",
                    help="只跑某一节：exp1/exp2/exp3/exp4/exp5/exp6")
    args = ap.parse_args()

    todo = args.only.split(",") if args.only else ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]
    if "exp1" in todo:
        exp1_single_head(seed=args.seed)
    if "exp2" in todo:
        exp2_scale_effect()
    if "exp3" in todo:
        exp3_multi_head(seed=args.seed, n_heads=args.h)
    if "exp4" in todo:
        exp4_positional_encoding()
    if "exp5" in todo:
        exp5_causal_mask()
    if "exp6" in todo:
        exp6_torch_check(n_heads=args.h, check_nn_mha=not args.no_nn_check)

    section("结论（对应论文第 3 节）")
    print("1. 注意力本质上是'可微的软检索'：用 query 和 key 算匹配分，softmax 变成权重，")
    print("   再对 value 做加权平均 —— 全部由矩阵乘 + softmax 组成，没有循环、没有卷积。")
    print("2. 除以 √d_k 是数值稳定性设计：点积的方差 ≈ d_k，不缩放会让 softmax 饱和。")
    print("3. 多头 = 在不同子空间里并行做检索，最后拼接融合；参数量与单头同量级，")
    print("   但能同时表达多种关系（论文表 3 行 A 的 24.9 → 25.8 BLEU）。")


if __name__ == "__main__":
    main()
