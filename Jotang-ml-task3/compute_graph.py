# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 3 · 实践 1：简单计算链

题目：对函数 f(x, y) = (xy + x²)²
    1. 把计算过程拆成若干简单步骤，并画出计算图；
    2. 写一段 NumPy 代码，按照计算图从后往前计算梯度；
    3. 使用 PyTorch 的 autograd 检查自己的结果是否一致。

本脚本做的事：
    * 取 x = 1.5, y = -2.0（y 取负数，中间量 u = xy + x² 是负的，
      正好检验“平方节点的导数 2u 带着负号”这种容易写错的地方），
      把前向拆成 4 个基本运算（平方 / 乘法 / 加法 / 平方），逐节点打印数值；
    * 从 f 出发按链式法则逐个节点回传，并显式体现 x 处“两条路径相加”（fan-out 求和）；
    * 用中心差分数值梯度做第三方验证（完全不依赖任何自动微分）；
    * 用 PyTorch 对同一条计算图做 autograd，逐项对比；
    * 再随机取 20 组 (x, y) 重复三方对比，避免只有一组数“碰巧对上”；
    * 存两张图：01_compute_graph.png（前向 / 反向计算图）
                02_grad_check.png（三方对比 + 等高线里的梯度方向）。

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python compute_graph.py
    2) conda run -n jotang-ml python compute_graph.py
    3) python compute_graph.py --x 2.0 --y 3.0
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
# 这是 Intel/LLVM 官方给出的开关，只影响“允许多份 OpenMP 共存”，不改变数值结果
# （本脚本的梯度另有“中心差分”做交叉验证，不依赖 MKL）。
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                                   # 不出窗口，直接存图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
LINE = "=" * 74


def section(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


# ==================================================================
# 一、前向：把 f(x, y) = (xy + x²)² 拆成 4 个基本运算
# ------------------------------------------------------------------
# 计算图（箭头方向 = 数据流动方向）：
#
#         ┌───────┐  a = x²  ┌───────┐
#      x ─┤   x   ├─────────►│   a   ├─┐
#         └───┬───┘          └───────┘ │   ┌───────┐        ┌───────┐
#             │                        ├──►│ u=a+b ├───────►│  f=u² │
#         ┌───▼───┐  b = x·y ┌───────┐ │   └───────┘        └───────┘
#      y ─┤   y   ├─────────►│   b   ├─┘
#         └───────┘          └───────┘
#
# 为什么这样拆：题目里的 f 用到了 d(x²)、d(x·y)、d(a+b)、d(u²) 四种导数，
# 拆到这一步每个节点都只含一种基本运算，求导时不需要再动脑子。
#
# 注意 a = x² 和 b = xy 都用了 x —— 这在反向传播里叫“扇出（fan-out）”，
# 意味着 x 处要把两条路径带回来的梯度**加起来**，而不是覆盖。
# ==================================================================
def forward_numpy(x, y):
    """前向计算，返回图上每个节点的值（顺便给反向传播当缓存用）。"""
    a = x ** 2          # 节点 a：x 的平方
    b = x * y           # 节点 b：x 乘以 y
    u = a + b           # 节点 u：加法
    f = u ** 2          # 节点 f：u 的平方（最终输出）
    return {"a": a, "b": b, "u": u, "f": f}


# ==================================================================
# 二、反向：从 f 出发，按计算图从后往前算梯度
# ------------------------------------------------------------------
# 反向传播 = 反复用链式法则：上游带下来的梯度 × 本节点局部导数。
# 每个基本运算的局部导数（就是它自己的求导公式）：
#
#   节点       前向公式     局部导数 ∂输出/∂输入
#   ─────────────────────────────────────────────
#   a = x²     a = x*x      ∂a/∂x = 2x
#   b = x·y    b = x*y      ∂b/∂x = y ,  ∂b/∂y = x
#   u = a+b    u = a+b      ∂u/∂a = 1 ,  ∂u/∂b = 1
#   f = u²     f = u*u      ∂f/∂u = 2u
#
# 反向顺序正好是前向的倒序：f → u → (a, b) → x, y。
# ==================================================================
def backward_numpy(x, y, cache):
    """从 f 出发按计算图倒着走，返回所有节点上的梯度（∂f/∂节点）。"""
    a, b, u = cache["a"], cache["b"], cache["u"]
    grads = {}

    # ---- 起点：梯度是一把“1”，表示 ∂f/∂f = 1 ----
    grads["f"] = 1.0

    # ---- 反向第 1 步：经过节点 f = u² ----
    local_df_du = 2.0 * u                       # 局部导数 ∂f/∂u = 2u
    grads["u"] = grads["f"] * local_df_du       # ∂f/∂u = ∂f/∂f · 2u

    # ---- 反向第 2 步：经过节点 u = a + b（加法把梯度原样分给两路）----
    grads["a"] = grads["u"] * 1.0               # ∂u/∂a = 1
    grads["b"] = grads["u"] * 1.0               # ∂u/∂b = 1

    # ---- 反向第 3 步：分别经过 a = x² 和 b = x·y ----
    grad_from_a = grads["a"] * (2.0 * x)        # ∂a/∂x = 2x
    grad_from_b = grads["b"] * y                # ∂b/∂x = y
    grads["y"] = grads["b"] * x                 # ∂b/∂y = x

    # ---- 汇聚（fan-out）：x 有两条出边，梯度相加 ----
    grads["x"] = grad_from_a + grad_from_b

    return grads, grad_from_a, grad_from_b


def closed_form(x, y):
    """直接用解析式验算：∂f/∂x = 2(xy+x²)(y+2x)，∂f/∂y = 2x(xy+x²)。"""
    grad_x = 2.0 * (x * y + x ** 2) * (y + 2.0 * x)
    grad_y = 2.0 * x * (x * y + x ** 2)
    return grad_x, grad_y


def f_scalar(x, y):
    """标量函数本体，给数值梯度（中心差分）用。"""
    return float((x * y + x ** 2) ** 2)


def numeric_grad(x, y, h=1e-6):
    """中心差分数值梯度：(f(x+h) - f(x-h)) / 2h，误差 O(h²)。"""
    gx = (f_scalar(x + h, y) - f_scalar(x - h, y)) / (2.0 * h)
    gy = (f_scalar(x, y + h) - f_scalar(x, y - h)) / (2.0 * h)
    return gx, gy


# ==================================================================
# 三、PyTorch autograd：对“同一条计算图”做自动微分
# ==================================================================
def torch_forward_backward(x, y, decomposed=True):
    """decomposed=True 时用和手工版完全相同的 4 步写法，方便一一对应。"""
    tx = torch.tensor(float(x), dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(float(y), dtype=torch.float64, requires_grad=True)

    if decomposed:
        a = tx ** 2                 # 节点 a
        b = tx * ty                 # 节点 b
        u = a + b                   # 节点 u
        f = u ** 2                  # 节点 f
    else:
        # 一行写完的版本，PyTorch 会自己拆成一模一样的图
        f = (tx * ty + tx ** 2) ** 2

    f.backward()
    return float(tx.grad), float(ty.grad), f


def graph_of(f, max_depth=4):
    """把 PyTorch 自己建好的反向图打印出来（证明它拆的步骤和我们一样）。"""
    lines = []

    def walk(node, depth):
        if node is None or depth > max_depth:
            return
        name = type(node).__name__
        lines.append("  " * depth + f"- {name}")
        for nxt, _ in getattr(node, "next_functions", ()):  # 一个节点的多个输入
            walk(nxt, depth + 1)

    walk(f.grad_fn, 0)
    return lines


# ==================================================================
# 四、画计算图
# ==================================================================
def border_point(cx, cy, w, h, tx, ty, pad=0.10):
    """求从节点中心指向 (tx, ty) 的射线与矩形边框的交点（让箭头贴着框）。"""
    hw, hh = w / 2.0 + pad, h / 2.0 + pad
    dx, dy = tx - cx, ty - cy
    if dx == 0.0 and dy == 0.0:
        return cx, cy
    sx = hw / abs(dx) if dx else float("inf")
    sy = hh / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def draw_node(ax, cx, cy, w, h, lines, edgecolor, facecolor, fontsize=9.0):
    box = FancyBboxPatch((cx - w / 2.0, cy - h / 2.0), w, h,
                         boxstyle="round,pad=0.10,rounding_size=0.16",
                         linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
            fontsize=fontsize, linespacing=1.55, zorder=3, color="#111111")


def draw_edge(ax, n1, n2, color, lw=1.6, label=None, lpos=(0.5, 0.0, 0.0),
              label_fontsize=8.5, label_color="#333333"):
    """在两个节点之间画箭头；label 放在边的 lpos = (比例 t, 位移 dx, 位移 dy) 处。"""
    (cx1, cy1, w1, h1) = n1
    (cx2, cy2, w2, h2) = n2
    x1, y1 = border_point(cx1, cy1, w1, h1, cx2, cy2)
    x2, y2 = border_point(cx2, cy2, w2, h2, cx1, cy1)
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=1))
    if label:
        t, dx, dy = lpos
        ax.text(x1 + (x2 - x1) * t + dx, y1 + (y2 - y1) * t + dy, label,
                ha="center", va="center", fontsize=label_fontsize, color=label_color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=4)


def plot_graph(x, y, cache, grads, gx_from_a, gx_from_b, outpath):
    a, b, u, f = cache["a"], cache["b"], cache["u"], cache["f"]
    # 每个节点的 (中心 x, 中心 y, 宽, 高)：两张子图共用同一套布局，方便左右对照
    nx = (1.00, 4.60, 2.10, 1.10)
    ny = (1.00, 1.80, 2.10, 1.10)
    na = (4.90, 5.15, 2.20, 1.00)
    nb = (4.90, 1.80, 2.20, 1.00)
    nu = (8.05, 3.50, 2.20, 1.00)
    nf = (11.55, 3.50, 2.00, 1.00)
    g = lambda v: f"{v:.4g}"                        # 输出保留 4 位有效数字，够看又不挤

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.8))
    for ax in axes:
        ax.set_xlim(-0.35, 12.75)
        ax.set_ylim(0.40, 6.30)
        ax.axis("off")

    # ---------------- 左：前向 ----------------
    ax = axes[0]
    ax.set_title(f"前向：把 f(x,y)=(xy+x²)² 拆成 4 个基本运算\n"
                 f"取 x = {x:g}, y = {y:g} → f = {g(f)}", fontsize=11.5)
    draw_node(ax, *nx, ["x", f"= {x:g}"], "#2f6f9f", "#e8f2fa")
    draw_node(ax, *ny, ["y", f"= {y:g}"], "#2f6f9f", "#e8f2fa")
    draw_node(ax, *na, ["a = x²", f"= {g(a)}"], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nb, ["b = x·y", f"= {g(b)}"], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nu, ["u = a + b", f"= {g(u)}"], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nf, ["f = u²", f"= {g(f)}"], "#a33b3b", "#fdecec")

    draw_edge(ax, nx, na, "#4a7ea8", label="a = x·x", lpos=(0.55, 0.0, 0.30))
    draw_edge(ax, nx, nb, "#4a7ea8", label="b = x·y", lpos=(0.55, 0.15, 0.16))
    draw_edge(ax, ny, nb, "#4a7ea8", label="b = x·y", lpos=(0.50, 0.0, 0.28))
    draw_edge(ax, na, nu, "#4a7ea8", label="+", label_fontsize=11, lpos=(0.5, 0.0, 0.20))
    draw_edge(ax, nb, nu, "#4a7ea8", label="+", label_fontsize=11, lpos=(0.5, 0.0, -0.20))
    draw_edge(ax, nu, nf, "#4a7ea8", label="u·u", lpos=(0.5, 0.0, 0.30))
    ax.text(6.35, 0.72, "数据从左流到右：每一条边只做一件小事，整条链合起来就是 (xy + x²)²",
            ha="center", fontsize=9.5, color="#555555")

    # ---------------- 右：反向 ----------------
    ax = axes[1]
    ax.set_title("反向：从 f 出发倒着走，每步乘上本节点的局部导数\n"
                 "红色箭头 = 梯度回传方向，边上的标签是局部导数，框里是流到该处的 ∂f/∂·",
                 fontsize=11.5)
    draw_node(ax, *nx, ["x", f"= {x:g}", f"∂f/∂x = {g(grads['x'])}"], "#b3451f", "#ffe9dc")
    draw_node(ax, *ny, ["y", f"= {y:g}", f"∂f/∂y = {g(grads['y'])}"], "#b3451f", "#ffe9dc")
    draw_node(ax, *na, ["a = x²", f"∂f/∂a = {g(grads['a'])}"], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nb, ["b = x·y", f"∂f/∂b = {g(grads['b'])}"], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nu, ["u = a + b", f"∂f/∂u = {g(grads['u'])}"], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nf, ["f = u²", "∂f/∂f = 1"], "#a33b3b", "#fdecec")

    draw_edge(ax, nf, nu, "#c0392b", label=f"∂f/∂u = 2u = {g(2 * u)}",
              lpos=(0.5, 0.0, 0.62))
    draw_edge(ax, nu, na, "#c0392b", label="∂u/∂a = 1", lpos=(0.5, 0.0, 0.22))
    draw_edge(ax, nu, nb, "#c0392b", label="∂u/∂b = 1", lpos=(0.5, 0.0, -0.22))
    draw_edge(ax, na, nx, "#c0392b", label=f"∂a/∂x = 2x = {g(2 * x)}",
              lpos=(0.5, 0.0, 0.32))
    draw_edge(ax, nb, nx, "#c0392b", label=f"∂b/∂x = y = {g(y)}", lpos=(0.5, 0.0, -0.30))
    draw_edge(ax, nb, ny, "#c0392b", label=f"∂b/∂y = x = {g(x)}", lpos=(0.5, 0.0, -0.28))
    ax.text(6.35, 0.72,
            f"关键：x 有两条出边（a、b），两条路径带回来的梯度必须相加 ——"
            f"∂f/∂x = ({g(gx_from_a)}) + ({g(gx_from_b)}) = {g(grads['x'])}",
            ha="center", fontsize=9.5, color="#8e2b1c")

    fig.suptitle("实践 1：简单计算链的计算图（前向 / 反向）", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"已保存计算图：{outpath}")


# ==================================================================
# 五、三方梯度对比图
# ==================================================================
def plot_check(x, y, cache, grads, num_grad, torch_grad, rand_err, outpath):
    fig = plt.figure(figsize=(14.0, 5.4))
    ax1 = fig.add_axes((0.06, 0.16, 0.36, 0.68))
    ax2 = fig.add_axes((0.50, 0.14, 0.47, 0.72))

    # ---- 左：三种来源的梯度柱状对比 ----
    labels = ["NumPy 按计算图手写", "中心差分（数值）", "PyTorch autograd"]
    gx_vals = [grads["x"], num_grad[0], torch_grad[0]]
    gy_vals = [grads["y"], num_grad[1], torch_grad[1]]
    pos = np.arange(3.0)
    w = 0.32
    b1 = ax1.bar(pos - w / 2, gx_vals, w, label="∂f/∂x", color="#2f6f9f")
    b2 = ax1.bar(pos + w / 2, gy_vals, w, label="∂f/∂y", color="#c0392b")
    for bars in (b1, b2):
        for rect in bars:
            v = rect.get_height()
            ax1.annotate(f"{v:.6f}", (rect.get_x() + rect.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 5 if v >= 0 else -13),
                         ha="center", fontsize=9)
    ax1.axhline(0.0, color="#888888", lw=0.9)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels, fontsize=9)
    err_here = max(abs(grads["x"] - torch_grad[0]), abs(grads["y"] - torch_grad[1]))
    ax1.set_title(f"∂f/∂x、∂f/∂y 的三种算法对比（x={x:g}, y={y:g}）\n"
                  f"手写反向 vs autograd：最大绝对误差 {err_here:.2e}\n"
                  f"再随机抽 20 组 (x, y)：最大绝对误差 {rand_err:.2e}",
                  fontsize=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    _all = gx_vals + gy_vals + [0.0]                # 一定要把 0 包进来，否则负值柱子会看不见
    _lo, _hi = min(_all), max(_all)
    _span = max(_hi - _lo, 1e-6)
    ax1.set_ylim(_lo - 0.15 * _span, _hi + 0.55 * _span)

    # ---- 右：等高线 + 梯度方向 ----
    xs = np.linspace(-3.2, 3.2, 400)
    ys = np.linspace(-4.9, 3.4, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (X * Y + X ** 2) ** 2
    levels = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    ax2.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.32)
    cs = ax2.contour(X, Y, Z, levels=levels, colors="#4a4a4a", linewidths=0.7)
    ax2.clabel(cs, inline=True, inline_spacing=1, fontsize=7, fmt="%g")
    ax2.plot(X, -X, color="#6f6f6f", lw=1.5, ls="--", zorder=3)   # xy + x² = 0，即 f = 0

    gx, gy = grads["x"], grads["y"]
    norm = float(np.hypot(gx, gy))
    scale = 1.15 / norm                        # 只把箭头长度归一化，方便看方向
    ax2.annotate("", xy=(x + gx * scale, y + gy * scale), xytext=(x, y),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.4))
    ax2.annotate("", xy=(x - gx * scale, y - gy * scale), xytext=(x, y),
                 arrowprops=dict(arrowstyle="-|>", color="#2f6f9f", lw=2.4, ls="--"))
    ax2.plot([x], [y], "o", color="black", ms=6, zorder=5)
    _wb = dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.88)
    ax2.text(x + 0.09, y - 0.26, f"({x:g}, {y:g})", fontsize=8.5, color="#222222",
             ha="left", va="top", bbox=_wb, zorder=6)
    ax2.text(x + gx * scale - 0.18, y + gy * scale + 0.30,
             f"$\\nabla f$ = ({gx:g}, {gy:g})", fontsize=8.5, color="#c0392b",
             ha="center", va="center", bbox=_wb, zorder=6)
    ax2.text(x - gx * scale + 0.20, y - gy * scale + 0.12, "$-\\nabla f$",
             fontsize=8.5, color="#2f6f9f", ha="left", va="bottom", bbox=_wb, zorder=6)
    ax2.text(-3.05, -4.75,
             "红实线箭头 $\\nabla f$：f 上升最快的方向（垂直于等高线）\n"
             "蓝虚箭头 $-\\nabla f$：梯度下降实际走的方向\n"
             "灰虚线：$xy + x^2 = 0$，即 f = 0 的那条谷底\n"
             "彩色填充：f 的等高线（越亮 f 越大）",
             fontsize=8, color="#333333", ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#cccccc", alpha=0.92))
    ax2.set_xlabel("x", fontsize=10)
    ax2.set_ylabel("y", fontsize=10)
    ax2.set_title("等高线图里的梯度：$\\nabla f$ 垂直于等高线、指向 f 变大的一侧", fontsize=11)
    ax2.grid(alpha=0.25, linestyle=":")
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-4.9, 3.4)

    fig.suptitle("实践 1：手写反向传播 vs 数值微分 vs PyTorch autograd", fontsize=13.5, y=0.97)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"已保存验证图：{outpath}")


# ==================================================================
# 六、主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="实践 1：简单计算链 (xy + x²)²")
    ap.add_argument("--x", type=float, default=1.5, help="输入 x（默认 1.5）")
    ap.add_argument("--y", type=float, default=-2.0, help="输入 y（默认 -2.0）")
    ap.add_argument("--outdir", type=str, default=HERE, help="图片输出目录")
    ap.add_argument("--n-random", type=int, default=20, help="随机抽查的组数")
    args = ap.parse_args()
    x, y = float(args.x), float(args.y)
    outdir = _os.path.abspath(args.outdir)
    _os.makedirs(outdir, exist_ok=True)

    section("任务 0：选定的 (x, y) 与函数 f(x, y) = (xy + x²)²")
    print(f"  x = {x:g} ,  y = {y:g}")
    print("  取 y 为负数是有意的：中间量 u = xy + x² 会变成负数，")
    print("  平方节点 f = u² 的局部导数 2u 就会带负号，能检验有没有写漏符号。")

    section("任务 1：前向 —— 把计算过程拆成 4 个简单步骤")
    cache = forward_numpy(x, y)
    a, b, u, f = cache["a"], cache["b"], cache["u"], cache["f"]
    print(f"  ① a = x² = ({x:g})² = {a:g}")
    print(f"  ② b = x·y = ({x:g}) × ({y:g}) = {b:g}")
    print(f"  ③ u = a + b = ({a:g}) + ({b:g}) = {u:g}      ← 题目里的 xy + x²")
    print(f"  ④ f = u² = ({u:g})² = {f:g}     ← 最终输出")
    print()
    print("  计算图结构（数据从左流到右）：")
    for line in [
        "    x ──► a = x²  ─┐",
        "                   ├──► u = a + b ──► f = u²",
        "    x ──► b = x·y ─┘",
        "    y ──► b = x·y ─┘",
    ]:
        print(line)

    section("任务 2：反向 —— 按计算图从后往前算梯度（纯 NumPy）")
    grads, gx_a, gx_b = backward_numpy(x, y, cache)
    print("  起点：∂f/∂f = 1（梯度是从输出往回流的，所以从 1 开始）")
    print()
    print("  反向第 1 步｜节点 f = u²")
    print(f"    局部导数 ∂f/∂u = 2u = 2 × ({u:g}) = {2 * u:g}   ← u 是负数，导数是负的")
    print(f"    ∂f/∂u = ∂f/∂f × 2u = 1 × {2 * u:g} = {grads['u']:g}")
    print()
    print("  反向第 2 步｜节点 u = a + b（加法的局部导数都是 1，梯度原样分给两路）")
    print(f"    ∂f/∂a = ∂f/∂u × ∂u/∂a = {grads['u']:g} × 1 = {grads['a']:g}")
    print(f"    ∂f/∂b = ∂f/∂u × ∂u/∂b = {grads['u']:g} × 1 = {grads['b']:g}")
    print()
    print("  反向第 3 步｜节点 a = x²（沿这条路径回到 x）")
    print(f"    局部导数 ∂a/∂x = 2x = 2 × {x:g} = {2 * x:g}")
    print(f"    这条路径带回 x 的梯度 = ∂f/∂a × ∂a/∂x = {grads['a']:g} × {2 * x:g} = {gx_a:g}")
    print()
    print("  反向第 3 步｜节点 b = x·y（乘法要区分两个输入，分别用对方的局部导数）")
    print(f"    局部导数 ∂b/∂x = y = {y:g}  →  这条路径带回 x 的梯度 = {grads['b']:g} × {y:g} = {gx_b:g}")
    print(f"    局部导数 ∂b/∂y = x = {x:g}  →  ∂f/∂y = {grads['b']:g} × {x:g} = {grads['y']:g}")
    print()
    print("  汇聚（扇出 fan-out 求和）：x 有两条出边，两条路径的梯度必须相加")
    print(f"    ∂f/∂x = {gx_a:g} + ({gx_b:g}) = {grads['x']:g}")
    print(f"    ∂f/∂y = {grads['y']:g}")
    print()
    cfx, cfy = closed_form(x, y)
    print("  用解析式验算一遍：")
    print(f"    ∂f/∂x = 2(xy + x²)(y + 2x) = 2 × ({u:g}) × ({y + 2 * x:g}) = {cfx:g}  {'✔' if abs(cfx - grads['x']) < 1e-12 else '✘'}")
    print(f"    ∂f/∂y = 2x(xy + x²)        = 2 × {x:g} × ({u:g})      = {cfy:g}  {'✔' if abs(cfy - grads['y']) < 1e-12 else '✘'}")

    section("任务 3：用 PyTorch autograd 检查是否一致")
    tx_grad, ty_grad, f_tensor = torch_forward_backward(x, y, decomposed=True)
    print(f"  用和手写版完全相同的 4 步写法建图（dtype=float64）：")
    print("    tx = torch.tensor(x, requires_grad=True) ; ty = ...")
    print("    a = tx**2 ; b = tx*ty ; u = a + b ; f = u**2 ; f.backward()")
    print()
    print(f"  PyTorch 得到的 f            = {float(f_tensor):.10f}   （手写：{f:.10f}）")
    print(f"  PyTorch 的 ∂f/∂x            = {tx_grad:.12f}")
    print(f"  手写 NumPy 的 ∂f/∂x         = {grads['x']:.12f}   绝对误差 {abs(tx_grad - grads['x']):.2e}")
    print(f"  PyTorch 的 ∂f/∂y            = {ty_grad:.12f}")
    print(f"  手写 NumPy 的 ∂f/∂y         = {grads['y']:.12f}   绝对误差 {abs(ty_grad - grads['y']):.2e}")
    print()
    print("  一行写完的版本 (tx*ty + tx**2)**2 结果相同：", end="")
    tx2, ty2, _ = torch_forward_backward(x, y, decomposed=False)
    print(f" ∂f/∂x = {tx2:.12f} , ∂f/∂y = {ty2:.12f}")
    print()
    print("  PyTorch 自己建出来的反向图（也就是它认为的计算图）：")
    for line in graph_of(f_tensor):
        print("   " + line)

    num_grad = numeric_grad(x, y)
    print()
    print("  再补一个和自动微分无关的第三方验证——中心差分：")
    print(f"    ∂f/∂x ≈ [f(x+h,y) - f(x-h,y)] / 2h = {num_grad[0]:.10f}  （与手写差 {abs(num_grad[0] - grads['x']):.2e}）")
    print(f"    ∂f/∂y ≈ [f(x,y+h) - f(x,y-h)] / 2h = {num_grad[1]:.10f}  （与手写差 {abs(num_grad[1] - grads['y']):.2e}）")
    print("    中心差分的误差 = 截断误差 O(h²) + 舍入误差 O(ε·|f|/h)，h=1e-6 时以后者为主，")
    print("    所以只能验到 1e-8 ~ 1e-10 的绝对精度；要验“最后一位”得靠解析式和 autograd。")

    section(f"任务 3（加强）：随机抽 {args.n_random} 组 (x, y) 重复三方对比")
    rng = np.random.default_rng(0)
    worst_manual_vs_torch = 0.0
    worst_manual_vs_numeric = 0.0
    print(f"  {'x':>9} {'y':>9} {'f':>14} {'∂f/∂x(手写)':>14} {'∂f/∂x(torch)':>14} "
          f"{'∂f/∂y(手写)':>14} {'∂f/∂y(torch)':>14} {'|Δ|':>9}")
    for _ in range(args.n_random):
        rx = float(rng.uniform(-3.0, 3.0))
        ry = float(rng.uniform(-3.0, 3.0))
        c2 = forward_numpy(rx, ry)
        g2, _, _ = backward_numpy(rx, ry, c2)
        tgx, tgy, _ = torch_forward_backward(rx, ry, decomposed=True)
        ngx, ngy = numeric_grad(rx, ry)
        e_torch = max(abs(g2["x"] - tgx), abs(g2["y"] - tgy))
        e_num = max(abs(g2["x"] - ngx), abs(g2["y"] - ngy))
        worst_manual_vs_torch = max(worst_manual_vs_torch, e_torch)
        worst_manual_vs_numeric = max(worst_manual_vs_numeric, e_num)
        print(f"  {rx:9.4f} {ry:9.4f} {c2['f']:14.6f} {g2['x']:14.6f} {tgx:14.6f} "
              f"{g2['y']:14.6f} {tgy:14.6f} {e_torch:9.2e}")
    print()
    print(f"  手写反向 vs autograd   最大绝对误差：{worst_manual_vs_torch:.2e}  "
          f"（两边是同一串浮点运算、同样的顺序，所以结果逐位一致）")
    print(f"  手写反向 vs 中心差分   最大绝对误差：{worst_manual_vs_numeric:.2e}  （数值微分的精度）")
    print(f"  结论：三者一致，手写反向传播是对的。（误差只来自浮点舍入和 h 的选取，"
          f"不是公式错。）") if worst_manual_vs_torch < 1e-12 else print("  警告：手写结果与 autograd 不一致，需要检查公式！")

    section("出图")
    plot_graph(x, y, cache, grads, gx_a, gx_b, _os.path.join(outdir, "01_compute_graph.png"))
    plot_check(x, y, cache, grads, num_grad, (tx_grad, ty_grad),
               worst_manual_vs_torch, _os.path.join(outdir, "02_grad_check.png"))


if __name__ == "__main__":
    main()
