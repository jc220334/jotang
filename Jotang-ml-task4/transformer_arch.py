# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 4 · 拓展 0：自己重画一张 Transformer 结构图

题目（任务第 1 条）：
    阅读论文 Figure 1，分别找出 Encoder 和 Decoder，参考原图，
    用自己的方式重新画一张简化 / 拓展的 Transformer 结构图。

本脚本做的事：
    * 用 matplotlib 把 Figure 1 重画一遍，但按"数据怎么流"来排布，并补上原图没写清楚的几处：
        - 三个注意力子层各自的 Q / K / V 分别来自哪里（原图只有连线，没有说明）；
        - 每个子层外面的残差连接到底连到哪两个点；
        - 掩码自注意力旁边的"只能看 ≤ t"提示；
        - 右侧一块说明：残差 / LayerNorm / FFN / 位置编码各自做什么；
        - 底部标注论文 base 模型的超参数：N=6、d_model=512、h=8、d_k=d_v=64、d_ff=2048。

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python transformer_arch.py
    2) conda run -n jotang-ml python transformer_arch.py
    3) python transformer_arch.py --out 00_transformer_arch.png

输出：00_transformer_arch.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))

# 配色：一眼能分出"注意力 / Add&Norm / FFN / 输出层 / 嵌入"
C_EMB = "#e9ebf2"
C_PE = "#fdf1d6"
C_ATT = "#d7e7f7"
C_AN = "#fbe1cd"
C_FFN = "#dff0e1"
C_OUT = "#eee0f6"
C_TXT = "#1a1a1a"
LINE = "=" * 74


def box(ax, x, y, w, h, text, fc, fs=9.6, ec="#4a4a4a", lw=1.1, style="round"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"{style},pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=C_TXT,
            zorder=3, linespacing=1.45)


def arrow(ax, p1, p2, color="#4a4a4a", lw=1.3, style="-|>", ms=12, zorder=1):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, zorder=zorder,
                                 shrinkA=0, shrinkB=0))


def stack(ax, x, y0, w, items, gap=0.20, fs=9.6):
    """从下往上码盒子；items = [(高度, 文字, 填充色, 字号), ...]，返回每个盒子的 (中心 y, 高)。"""
    y = y0
    placed = []
    for it in items:
        h, text, fc = it[0], it[1], it[2]
        fontsize = it[3] if len(it) > 3 else fs
        y += h / 2
        box(ax, x, y, w, h, text, fc, fs=fontsize)
        placed.append((y, h))
        y += h / 2
        # 箭头：连到下一个盒子
        y += gap
    for (yc, hc), (yn, hn) in zip(placed[:-1], placed[1:]):
        arrow(ax, (x, yc + hc / 2), (x, yn - hn / 2))
    return placed


def residual(ax, x_left, y_from, y_to, color="#c0392b", rad=0.55, label=None):
    """残差连接：从子层的入口绕过子层，接到它的 Add & Norm 上。"""
    ax.add_patch(FancyArrowPatch((x_left, y_from), (x_left, y_to),
                                 arrowstyle="-|>", mutation_scale=11, color=color,
                                 lw=1.2, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(x_left - 0.34, (y_from + y_to) / 2, label, fontsize=7.8, color=color,
                rotation=90, ha="center", va="center")


def main():
    ap = argparse.ArgumentParser(description="重画 Transformer 结构图")
    ap.add_argument("--out", type=str, default="00_transformer_arch.png")
    ap.add_argument("--n-layers", type=int, default=6, help="堆叠层数（论文 base = 6）")
    args = ap.parse_args()

    N = args.n_layers
    fig, ax = plt.subplots(figsize=(16.6, 11.4))
    ax.set_xlim(0, 17.2)
    ax.set_ylim(0, 11.4)
    ax.axis("off")
    ax.set_title("Transformer 结构图（按论文 Figure 1 重画，并补上 Q/K/V 的来源与残差走向）",
                 fontsize=14, y=0.985)

    # ---------------- Encoder（左） ----------------
    xe, we = 3.6, 3.3
    items_enc = [
        (0.52, "Input Embedding（词嵌入）", C_EMB),
        (0.52, "+ Positional Encoding  ⊕", C_PE),
        (0.92, "Multi-Head\nSelf-Attention\n自注意力", C_ATT, 9.2),
        (0.46, "Add & Norm", C_AN),
        (0.92, "Position-wise\nFeed Forward\n两层全连接 512→2048→512", C_FFN, 9.0),
        (0.46, "Add & Norm", C_AN),
    ]
    enc = stack(ax, xe, 1.05, we, items_enc)
    ax.text(xe, 0.72, "输入序列 x1 … xn（源语言，如英文）", ha="center", fontsize=10,
            color="#555555")
    arrow(ax, (xe, 0.82), (xe, enc[0][0] - enc[0][1] / 2))

    # 残差连接（encoder 两个子层）
    residual(ax, xe - we / 2 - 0.08, enc[1][0] + enc[1][1] / 2 + 0.02, enc[3][0],
             label="残差连接")
    residual(ax, xe - we / 2 - 0.08, enc[3][0] + enc[3][1] / 2 + 0.02, enc[5][0])

    # N× 括号
    y_top, y_bot = enc[5][0] + enc[5][1] / 2, enc[2][0] - enc[2][1] / 2
    xb = xe + we / 2 + 0.18
    ax.plot([xb, xb + 0.22, xb + 0.22, xb], [y_top, y_top, y_bot, y_bot],
            color="#333333", lw=1.2)
    ax.text(xb + 0.30, (y_top + y_bot) / 2, f"N× = {N} 层", fontsize=10, rotation=90,
            va="center", color="#333333")
    ax.text(xe, y_top + 1.05, "Encoder（编码器）", ha="center", fontsize=12.5,
            color="#1f4e79", fontweight="bold")

    # 位置编码说明（指向 encoder 的 ⊕）
    y_pe_e = enc[1][0]
    box(ax, 0.95, y_pe_e, 1.55, 0.78, "Positional\nEncoding\n正弦 / 余弦", C_PE, fs=8.4)
    arrow(ax, (1.72, y_pe_e), (xe - we / 2, y_pe_e), color="#b0851a", lw=1.1)

    # ---------------- Decoder（右） ----------------
    xd, wd = 10.0, 3.9
    items_dec = [
        (0.52, "Output Embedding（词嵌入）", C_EMB),
        (0.52, "+ Positional Encoding  ⊕", C_PE),
        (0.92, "Masked Multi-Head\nSelf-Attention\n掩码自注意力（只能看 ≤ t）", C_ATT, 8.8),
        (0.46, "Add & Norm", C_AN),
        (0.92, "Multi-Head Attention\n交叉注意力\nQ 来自 decoder，K/V 来自 encoder", C_ATT, 8.6),
        (0.46, "Add & Norm", C_AN),
        (0.92, "Position-wise\nFeed Forward\n两层全连接 512→2048→512", C_FFN, 9.0),
        (0.46, "Add & Norm", C_AN),
        (0.44, "Linear（投影回词表大小）", C_OUT, 8.8),
        (0.44, "Softmax", C_OUT, 9.2),
    ]
    dec = stack(ax, xd, 1.05, wd, items_dec)
    ax.text(xd, 0.72, "输出序列 y1 … ym（右移一位 shifted right）", ha="center", fontsize=10,
            color="#555555")
    arrow(ax, (xd, 0.82), (xd, dec[0][0] - dec[0][1] / 2))
    ax.text(xd, dec[9][0] + dec[9][1] / 2 + 0.42, "输出概率分布（下一个词）", ha="center",
            fontsize=10, color="#555555")

    # 残差连接（decoder 三个子层）
    residual(ax, xd - wd / 2 - 0.10, dec[1][0] + dec[1][1] / 2 + 0.02, dec[3][0],
             label="残差连接")
    residual(ax, xd - wd / 2 - 0.10, dec[3][0] + dec[3][1] / 2 + 0.02, dec[5][0])
    residual(ax, xd - wd / 2 - 0.10, dec[5][0] + dec[5][1] / 2 + 0.02, dec[7][0])

    # N× 括号
    y_top_d, y_bot_d = dec[7][0] + dec[7][1] / 2, dec[2][0] - dec[2][1] / 2
    xbd = xd + wd / 2 + 0.18
    ax.plot([xbd, xbd + 0.22, xbd + 0.22, xbd], [y_top_d, y_top_d, y_bot_d, y_bot_d],
            color="#333333", lw=1.2)
    ax.text(xbd + 0.30, (y_top_d + y_bot_d) / 2, f"N× = {N} 层", fontsize=10, rotation=90,
            va="center", color="#333333")
    ax.text(xd, dec[9][0] + dec[9][1] / 2 + 1.05, "Decoder（解码器，逐步生成）", ha="center",
            fontsize=12.5, color="#7c3f1d", fontweight="bold")

    # 位置编码说明（指向 decoder 的 ⊕）
    y_pe_d = dec[1][0]
    arrow(ax, (xd + wd / 2 + 0.06, y_pe_d), (xd + wd / 2 + 0.55, y_pe_d),
          color="#b0851a", lw=1.1)
    ax.text(xd + wd / 2 + 0.62, y_pe_d, "同样加一次\n位置编码", fontsize=8.4, color="#8a6a12",
            va="center")

    # ---------------- Encoder → Decoder 的 K/V ----------------
    y_src = enc[5][0] + enc[5][1] / 2
    y_cross = dec[4][0]
    xroute = xd - wd / 2 - 0.62
    ax.plot([xe, xe, xroute, xroute], [y_src, y_src + 0.30, y_src + 0.30, y_cross],
            color="#1f6f4a", lw=1.5, zorder=1)
    arrow(ax, (xroute, y_cross), (xd - wd / 2, y_cross), color="#1f6f4a", lw=1.5)
    ax.text(xroute - 0.10, (y_src + 0.30 + y_cross) / 2, "K、V（编码器输出）", fontsize=8.6,
            rotation=90, va="center", ha="center", color="#1f6f4a")
    ax.text((xe + xroute) / 2, y_src + 0.42, "编码器输出：整句的向量表示（memory）",
            fontsize=8.6, color="#1f6f4a", ha="center", va="bottom")

    # ---------------- 右侧说明 ----------------
    x0 = 13.75
    ax.text(x0, 10.50, "三种注意力的 Q / K / V 从哪来", fontsize=10.4, fontweight="bold",
            color="#1a1a1a")
    ax.text(x0, 10.05,
            "① Encoder 自注意力\n"
            "    Q = K = V = 上一层 encoder 输出\n"
            "    每个词都能看整句（双向）\n"
            "② Decoder 掩码自注意力\n"
            "    Q = K = V = 上一层 decoder 输出\n"
            "    只能看 ≤ 当前词（右侧被 -∞ 堵住）\n"
            "③ 交叉注意力\n"
            "    Q 来自 decoder 上一层\n"
            "    K = V = encoder 的输出（对齐源句）",
            fontsize=8.5, va="top", linespacing=1.6, color="#2a2a2a")

    ax.text(x0, 7.30, "容易糊的四个地方", fontsize=10.4, fontweight="bold", color="#1a1a1a")
    ax.text(x0, 6.88,
            "• 残差连接：绕过子层的那条边\n"
            "  输出 = LayerNorm(x + Sublayer(x))\n"
            "  梯度可沿它直接回流到浅层\n"
            "• LayerNorm：在特征维上归一化\n"
            "  （和 BatchNorm 不是一个方向）\n"
            "• Feed Forward：逐位置的两层 MLP\n"
            "  注意力负责「取信息」，\n"
            "  FFN 负责「加工信息」，交替出现\n"
            "• 位置编码：全模型唯一的顺序来源\n"
            "  （3.5 节），加在词嵌入上",
            fontsize=8.5, va="top", linespacing=1.6, color="#2a2a2a")

    ax.text(x0, 4.25, "论文 base 模型超参数", fontsize=10.4, fontweight="bold", color="#1a1a1a")
    ax.text(x0, 3.83,
            "N = 6 层，d_model = 512\n"
            "h = 8 个头，d_k = d_v = 64\n"
            "d_ff = 2048（内层 4×）\n"
            "dropout = 0.1，label smoothing = 0.1\n"
            "8 张 P100 训练 12 小时 / 100k 步",
            fontsize=8.5, va="top", linespacing=1.6, color="#2a2a2a")

    # ---------------- 图例 ----------------
    legend = [("注意力子层", C_ATT), ("Add & Norm（残差 + LayerNorm）", C_AN),
              ("逐位置前馈 FFN", C_FFN), ("嵌入 / 位置编码", C_PE),
              ("输出层 Linear+Softmax", C_OUT)]
    xl = 0.6
    for text, fc in legend:
        ax.add_patch(Rectangle((xl, 0.16), 0.26, 0.20, fc=fc, ec="#4a4a4a", lw=1.0))
        ax.text(xl + 0.34, 0.26, text, fontsize=8.4, va="center")
        xl += 0.50 + 0.135 * len(text)

    ax.text(0.6, 0.54,
            "读图提示：从下往上读 = 前向一次；Decoder 比 Encoder 多一个子层（交叉注意力），"
            "这就是翻译任务里「对齐源句」的那一步。",
            fontsize=9.0, color="#555555")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = _os.path.join(HERE, args.out)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(LINE)
    print("已保存 " + path)
    print(LINE)
    print("图的层次（从下往上）")
    print("  Encoder：Input Embedding → (+PE) → Multi-Head Self-Attention → Add&Norm")
    print("           → Feed Forward → Add&Norm        （共 N=6 层重复）")
    print("  Decoder：Output Embedding → (+PE) → Masked Self-Attention → Add&Norm")
    print("           → Cross-Attention(K,V 来自 encoder) → Add&Norm")
    print("           → Feed Forward → Add&Norm → Linear → Softmax")


if __name__ == "__main__":
    main()
