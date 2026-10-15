# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 3 · 实践 2：神经网络的梯度

题目：从 Task 1 的 make_moons 数据里取一个样本（再换成一个很小的 batch），
      构造两层二分类网络：  X → Linear → ReLU → Linear → Sigmoid
      只做一次前向 + 一次反向，不更新参数、不训练模型：
        1. 自己定隐藏层神经元数量，初始化 W1, b1, W2, b2；
        2. 用 NumPy 完成 Linear / ReLU / Sigmoid / 二元交叉熵损失的前向计算；
        3. 画出从输入到 loss 的计算图，按链式法则从后往前算每个参数的梯度；
        4. 用同样的输入和参数在 PyTorch 里重做一次前向，靠 autograd 拿梯度；
        5. 把 NumPy 手算梯度和 PyTorch 自动梯度逐项比较：形状、数值、最大绝对误差。

本脚本做的事：
    * 用和 Task 1 完全相同的方式生成并预处理 make_moons（同一 seed / 同一 StandardScaler），
      取第 1 个训练样本当 N=1 的例子；
    * 打印 X / W1 / b1 / Z1 / A1 / W2 / b2 / Z2 / P / L 以及它们梯度的形状；
    * 手写完整链式法则（Sigmoid 的导数 + BCE 的导数分开算），再和"p - t 融合写法"对照；
    * 用 PyTorch autograd 逐项比对，并补一个完全不靠自动微分的中心差分数值梯度做第三方验证；
    * 换成 N=8 的小 batch 再跑一遍，逐个说明哪些公式、哪些形状变了；
    * 故意写出三种典型错误（sum 当成 mean / 矩阵转置写反 / 忘记对 batch 求和），
      看它们分别长什么样，方便以后自己排查；
    * 存三张图：03_nn_compute_graph.png（计算图：前向带形状、反向带公式）
                04_nn_grad_check.png（逐项梯度对比 + 误差 + 典型错误诊断）
                05_nn_batch_effect.png（单个样本 vs 小 batch：形状与 1/N 因子）。

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python nn_backprop.py
    2) conda run -n jotang-ml python nn_backprop.py
    3) python nn_backprop.py --n-hidden 3 --batch-size 8
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

# 控制台编码兜底：Windows 的 cmd/PowerShell 默认是 GBK(936)，直接 print "∂" "σ" "∇"
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
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")                                   # 不出窗口，直接存图
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = _os.path.dirname(_os.path.abspath(__file__))
LINE = "=" * 78
SEED = 42                                               # 和 Task 1 用同一个种子


def section(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def shp(a):
    """把数组 / 标量的形状写成 '(2, 3)' 或 '()'，方便打印和画图。"""
    if a is None:
        return "-"
    if isinstance(a, (tuple, list)):                      # 直接给形状元组时（比如 a.shape）
        return "(" + ", ".join(str(d) for d in a) + ")" if len(a) else "()"
    if np.ndim(a) == 0:
        return "()"
    return "(" + ", ".join(str(d) for d in np.shape(a)) + ")"


# ==================================================================
# 一、数据：直接用 Task 1 的 make_moons（同样本、同划分、同标准化）
# ------------------------------------------------------------------
# Task 1 里是这样建的：make_moons(n_samples=1000, noise=0.2, random_state=42)，
# 再 train_test_split 成 训练/验证/测试，StandardScaler 只在训练集上 fit。
# 这里原样复用那几行，保证“样本”就是 Task 1 见过的那些点。
# ==================================================================
def load_single_and_batch(sample_index=0, batch_size=8):
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                             random_state=SEED, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2,
                                             random_state=SEED, stratify=y_tr)
    scaler = StandardScaler().fit(X_tr)                      # 只在训练集上 fit（避免数据泄漏）
    X_tr_s = scaler.transform(X_tr).astype(np.float64)

    i = int(sample_index) % len(X_tr_s)
    X1 = X_tr_s[i:i + 1]                                     # 形状 (1, 2)
    t1 = y_tr[i:i + 1].astype(np.float64).reshape(1, 1)      # 形状 (1, 1)
    Xb = X_tr_s[:batch_size]                                 # 形状 (N, 2)
    tb = y_tr[:batch_size].astype(np.float64).reshape(-1, 1)  # 形状 (N, 1)
    return (X1, t1), (Xb, tb), (X_tr_s, y_tr.astype(np.float64))


# ==================================================================
# 二、参数初始化（自己定隐藏层大小）与“为什么要这样定形状”
# ------------------------------------------------------------------
# 网络：X → Linear → ReLU → Linear → Sigmoid → loss
#
#   记号      含义                    形状（N 个样本，输入 2 维，隐藏 H 个神经元）
#   ─────────────────────────────────────────────────────────────
#   X        输入                    (N, 2)
#   W1       第一层权重（列 = 神经元） (2, H)
#   b1       第一层偏置              (H,)
#   Z1 = XW1 + b1  第一层的“logit”    (N, H)
#   A1 = relu(Z1)  隐藏层激活         (N, H)
#   W2       第二层权重              (H, 1)
#   b2       第二层偏置              (1,)
#   Z2 = A1W2 + b2 输出 logit         (N, 1)
#   P  = σ(Z2)     预测概率（0~1）    (N, 1)
#   T        真实标签                (N, 1)
#   L        BCE 损失（标量）         ()
#
# 为什么权重是这个朝向：PyTorch 的 nn.Linear(in, out) 里 weight 的形状就是 (out, in)，
# 它内部算 x @ W.T。我们手写时用 X @ W1，所以 W1 直接写成 (in, out) = (2, H)，
# 这样每一列对应一个神经元的全部输入权重，写回 PyTorch 时要转置一次。
#
# 隐藏层故意取很小（默认 H = 3）：题目说“难度稍大，先只用一个样本、隐藏层很小”，
# 3 个神经元 = 2*3+3+3*1+1 = 13 个参数，13 个梯度可以一个一个盯到底。
# ==================================================================
def init_params(n_in=2, n_hidden=3, n_out=1, seed=0):
    rng = np.random.default_rng(seed)
    # 标准初始化：W ~ N(0, sqrt(2/fan_in))，这里 fan_in 是输入个数
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out))
    b2 = np.zeros(n_out)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ==================================================================
# 三、NumPy 前向：把 X → Linear → ReLU → Linear → Sigmoid → BCE 一行行摊开
# ==================================================================
def relu(z):
    return np.maximum(z, 0.0)


def sigmoid(z):
    # 稳定写法：z 很负时 exp(-z) 会溢出，所以先按符号拆分（本题数值不大，但习惯要养好）
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def forward_numpy(params, X, T):
    """前向计算，返回 (loss, cache)。cache 里存的是反向传播要用的中间量。"""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]

    Z1 = X @ W1 + b1                    # Linear 1：(N,2)@(2,H) + (H,)  → (N,H)（b1 靠广播）
    A1 = relu(Z1)                       # ReLU：逐元素，形状不变 (N,H)
    Z2 = A1 @ W2 + b2                   # Linear 2：(N,H)@(H,1) + (1,) → (N,1)
    P = sigmoid(Z2)                     # Sigmoid：逐元素 → (N,1)，值域 (0,1)

    eps = 1e-12                         # 只防 log(0) 报警，正常不会触发
    Pc = np.clip(P, eps, 1.0 - eps)
    per_sample = -(T * np.log(Pc) + (1.0 - T) * np.log(1.0 - Pc))     # (N,1)
    L = float(np.mean(per_sample))                                     # 标量，对 batch 取平均

    cache = {"X": X, "T": T, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P,
             "N": N, "per_sample": per_sample}
    return L, cache


def loss_numpy(params, X, T):
    """只要 loss 值（数值梯度用）。"""
    return forward_numpy(params, X, T)[0]


# ==================================================================
# 四、NumPy 反向：按链式法则从 loss 倒着走回每个参数
# ------------------------------------------------------------------
# 链式法则到每一层的“局部导数表”：
#   节点                    前向                局部导数（对自己的输入求导）
#   ───────────────────────────────────────────────────────────────
#   Z1 = X W1 + b1          Linear              ∂Z1/∂W1 = Xᵀ  ,  ∂Z1/∂X = W1ᵀ ,  ∂Z1/∂b1 = 1
#   A1 = relu(Z1)           ReLU                ∂A1/∂Z1 = 1{Z1 > 0}（逐元素）
#   Z2 = A1 W2 + b2         Linear              ∂Z2/∂W2 = A1ᵀ ,  ∂Z2/∂A1 = W2ᵀ ,  ∂Z2/∂b2 = 1
#   P  = σ(Z2)              Sigmoid             ∂P/∂Z2 = P(1-P)（逐元素）
#   L  = BCE(P, T)          BCE                 ∂L/∂P = -(1/N)[T/P - (1-T)/(1-P)]
#   L  = mean(损失)          求平均              ∂L/∂每个样本损失 = 1/N   ← 1/N 就是从这里来的
#
# 关键结论（“趁热打铁”第 3 问）：Sigmoid 和 BCE 的导数相乘时 P(1-P) 会约掉：
#   ∂L/∂Z2 = ∂L/∂P ⊙ P(1-P) = (1/N)(P - T)
# 也就是“越自信越错、梯度越大”，一个减法就完事——框架里的 BCEWithLogitsLoss 用的就是它。
# 下面 backward_numpy 把两条路都实现出来做个对照：
#   chain=True   完整链式法则（σ' 和 BCE 的导数分开算）
#   chain=False  融合写法 dZ2 = (P - T)/N
# ==================================================================
def backward_numpy(params, cache, fused=True):
    X, T = cache["X"], cache["T"]
    Z1, A1, Z2, P = cache["Z1"], cache["A1"], cache["Z2"], cache["P"]
    N = cache["N"]
    W1, W2 = params["W1"], params["W2"]

    if fused:
        # 融合写法：Sigmoid + BCE 的导数先代数化简，一步得到 logit 的梯度
        dZ2 = (P - T) / N                        # (N,1)
    else:
        # 完整写法：一步一步来，最后结果应该完全一样（可用来检查有没有漏项）
        dL_dP = -(1.0 / N) * (T / P - (1.0 - T) / (1.0 - P))   # BCE 对 P 求导   (N,1)
        dP_dZ2 = P * (1.0 - P)                                  # Sigmoid 的导数 (N,1)
        dZ2 = dL_dP * dP_dZ2                                    # (N,1)

    A1T = A1.T                                   # (H,N)
    XT = X.T                                     # (2,N)

    dW2 = A1T @ dZ2                              # (H,N)@(N,1) → (H,1)，和 W2 同形状
    db2 = dZ2.sum(axis=0)                        # (1,) 把 batch 维“挤掉” → 和 b2 同形状
    dA1 = dZ2 @ W2.T                             # (N,1)@(1,H) → (N,H)，和 A1 同形状
    dZ1 = dA1 * (Z1 > 0.0)                       # ReLU 的导数：Z1<=0 的位置梯度直接置 0 → (N,H)
    dW1 = XT @ dZ1                               # (2,N)@(N,H) → (2,H)，和 W1 同形状
    db1 = dZ1.sum(axis=0)                        # (H,)，和 b1 同形状

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    mids = {"dL_dP": None if fused else dL_dP, "dP_dZ2": None if fused else dP_dZ2,
            "dZ2": dZ2, "dA1": dA1, "dZ1": dZ1}
    return grads, mids


# ==================================================================
# 五、数值梯度（中心差分）——完全不依赖自动微分的第三方验证
# ==================================================================
def numeric_grad(params, X, T, h=1e-6):
    """对 13 个参数逐个做中心差分：(L(p+h) - L(p-h)) / 2h，误差 O(h²)+舍入。"""
    grads = {}
    for k, P in params.items():
        g = np.zeros_like(P, dtype=np.float64)
        flat = P.reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + h
            lp = loss_numpy(params, X, T)
            flat[i] = old - h
            lm = loss_numpy(params, X, T)
            flat[i] = old
            g.reshape(-1)[i] = (lp - lm) / (2.0 * h)
        grads[k] = g
    return grads


# ==================================================================
# 六、PyTorch autograd：同一组参数、同一个前向，让框架自己求导
# ==================================================================
def torch_forward_backward(params, X, T, fused_loss=True):
    """把 NumPy 的 W1/b1/W2/b2 原样搬进 torch，跑一次前向 + backward。

    fused_loss=True  → F.binary_cross_entropy(σ(Z2), T)（sigmoid 与 BCE 分成两步，
                       和手写图完全对应）
    fused_loss=False → F.binary_cross_entropy_with_logits(Z2, T)（框架里的融合实现）
    两者梯度应当一致（后者在极端数值下更稳）。
    """
    tX = torch.tensor(X, dtype=torch.float64)
    tT = torch.tensor(T, dtype=torch.float64)
    tW1 = torch.tensor(params["W1"], dtype=torch.float64, requires_grad=True)
    tb1 = torch.tensor(params["b1"], dtype=torch.float64, requires_grad=True)
    tW2 = torch.tensor(params["W2"], dtype=torch.float64, requires_grad=True)
    tb2 = torch.tensor(params["b2"], dtype=torch.float64, requires_grad=True)

    tZ1 = tX @ tW1 + tb1
    tA1 = torch.relu(tZ1)
    tZ2 = tA1 @ tW2 + tb2
    if fused_loss:
        tP = torch.sigmoid(tZ2)
        tL = F.binary_cross_entropy(tP, tT)              # reduction='mean'（默认）
    else:
        tL = F.binary_cross_entropy_with_logits(tZ2, tT)
    tL.backward()
    grads = {"W1": tW1.grad.numpy(), "b1": tb1.grad.numpy(),
             "W2": tW2.grad.numpy(), "b2": tb2.grad.numpy()}
    mids = {"Z1": tZ1.detach().numpy(), "A1": tA1.detach().numpy(),
            "Z2": tZ2.detach().numpy(),
            "P": torch.sigmoid(tZ2).detach().numpy()}
    return float(tL), grads, mids


def graph_lines(f, max_depth=8):
    """把 autograd 自己建的反向图打出来，看它拆的步骤和我们画的是不是同一张图。"""
    lines = []

    def walk(node, depth):
        if node is None or depth > max_depth:
            return
        lines.append("  " * depth + "- " + type(node).__name__)
        for nxt, _ in getattr(node, "next_functions", ()):
            walk(nxt, depth + 1)

    walk(f.grad_fn, 0)
    return lines


def torch_graph_lines(params, X, T):
    """单独再建一次图，专门用来打印 autograd 关心的结构。"""
    tX = torch.tensor(X, dtype=torch.float64)
    tT = torch.tensor(T, dtype=torch.float64)
    tW1 = torch.tensor(params["W1"], dtype=torch.float64, requires_grad=True)
    tb1 = torch.tensor(params["b1"], dtype=torch.float64, requires_grad=True)
    tW2 = torch.tensor(params["W2"], dtype=torch.float64, requires_grad=True)
    tb2 = torch.tensor(params["b2"], dtype=torch.float64, requires_grad=True)
    tZ1 = tX @ tW1 + tb1
    tA1 = torch.relu(tZ1)
    tZ2 = tA1 @ tW2 + tb2
    tP = torch.sigmoid(tZ2)
    tL = F.binary_cross_entropy(tP, tT)
    return graph_lines(tL)


# ==================================================================
# 七、比较与打印
# ==================================================================
def compare(grads_np, grads_torch):
    """逐项比较，返回每项的最大绝对误差和总体最大误差。"""
    rows = []
    worst = 0.0
    for k in ("W1", "b1", "W2", "b2"):
        a, b = grads_np[k], grads_torch[k]
        e = float(np.max(np.abs(a - b)))
        worst = max(worst, e)
        rows.append((k, a.shape, a, b, e))
    return rows, worst


def print_params(params):
    for k in ("W1", "b1", "W2", "b2"):
        v = params[k]
        print(f"    {k:3s} shape = {shp(v):8s}  值 = {np.array2string(v, precision=6, suppress_small=True)}")


def print_forward_chain(params, X, T, cache, L):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N = X.shape[0]
    print("  前向：X → Linear → ReLU → Linear → Sigmoid → BCE，一步一个形状")
    print(f"    ① Z1 = X @ W1 + b1   {shp(X)} @ {shp(W1)} + {shp(b1)} → {shp(cache['Z1'])}"
          f"   （b1 靠广播加到每一行）")
    print(f"    ② A1 = relu(Z1)      {shp(cache['Z1'])} → {shp(cache['A1'])}"
          f"   （{int(np.sum(cache['Z1'] <= 0))} 个元素被置 0）")
    print(f"    ③ Z2 = A1 @ W2 + b2  {shp(cache['A1'])} @ {shp(W2)} + {shp(b2)} → {shp(cache['Z2'])}")
    print(f"    ④ P  = sigmoid(Z2)   {shp(cache['Z2'])} → {shp(cache['P'])}"
          f"   （值域 (0,1)）")
    print(f"    ⑤ L  = BCE(P, T) 再对 {N} 个样本取平均 → {shp(L)}（标量）= {L:.10f}")
    dead = cache["Z1"] <= 0.0
    if dead.any():
        if N == 1:
            units = ", ".join(f"第 {j + 1} 个（下标 {j}）"
                              for j in range(cache["Z1"].shape[1]) if dead[0, j])
            print(f"       说明：隐藏层 {units} 神经元在这个样本上 z1 ≤ 0，ReLU 输出 0，")
            print("       反向时 ∂A1/∂Z1 = 1{z1 > 0} = 0，所以它那一列的梯度会**正好是 0**。")
            print("       这是 ReLU 的正常行为（不是 bug），也是它“稀疏”的来源。")
        else:
            print(f"       说明：{int(dead.sum())} / {dead.size} 个 Z1 元素被 ReLU 置 0，"
                  f"它们对应的梯度也一律是 0。")
    if N == 1:
        print("       注意：N = 1 时“取平均”和“求和”结果一样，所以 1/N 这个因子")
        print("       在单个样本上看不出区别；换成 batch 才会暴露，见第 7 节。")


def print_backward_chain(params, cache, grads, mids):
    P, T, Z1, N = cache["P"], cache["T"], cache["Z1"], cache["N"]
    print("  反向：从 loss 倒着回到四个参数，每一步都写清形状")
    print(f"    ⑥ dZ2 = ∂L/∂Z2 = (P - T)/N     {shp(P)} → {shp(mids['dZ2'])}"
          f"   （P 与 T 相减，Sigmoid 与 BCE 的导数正好约掉 P(1-P)）")
    if mids["dL_dP"] is not None:
        print(f"       分开算也一样：∂L/∂P = -(1/N)[T/P-(1-T)/(1-P)] {shp(mids['dL_dP'])}"
              f" × ∂P/∂Z2 = P(1-P) {shp(mids['dP_dZ2'])} → {shp(mids['dL_dP'] * mids['dP_dZ2'])}")
        print(f"       两条路结果最大绝对差 = "
              f"{float(np.max(np.abs(mids['dL_dP'] * mids['dP_dZ2'] - mids['dZ2']))):.2e}")
    print(f"    ⑦ dW2 = A1ᵀ @ dZ2 = {shp(cache['A1'].T)} @ {shp(mids['dZ2'])} → {shp(grads['W2'])}"
          f"  （和 W2 同形状 {shp(params['W2'])}）")
    print(f"       db2 = dZ2.sum(axis=0)                     {shp(mids['dZ2'])} → {shp(grads['b2'])}"
          f"  （batch 维求和，形状回到 {shp(params['b2'])}）")
    print(f"    ⑧ dA1 = dZ2 @ W2.T = {shp(mids['dZ2'])} @ {shp(params['W2'].T)} → {shp(mids['dA1'])}")
    print(f"    ⑨ dZ1 = dA1 ⊙ 1{{Z1>0}}                     {shp(mids['dA1'])} → {shp(mids['dZ1'])}"
          f"  （ReLU 把负的那一半梯度清零）")
    print(f"    ⑩ dW1 = Xᵀ @ dZ1 = {shp(cache['X'].T)} @ {shp(mids['dZ1'])} → {shp(grads['W1'])}"
          f"  （和 W1 同形状 {shp(params['W1'])}）")
    print(f"       db1 = dZ1.sum(axis=0)                     {shp(mids['dZ1'])} → {shp(grads['b1'])}")


# ==================================================================
# 八、画图
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


def draw_node(ax, cx, cy, w, h, lines, edgecolor, facecolor, fontsize=8.6):
    box = FancyBboxPatch((cx - w / 2.0, cy - h / 2.0), w, h,
                         boxstyle="round,pad=0.10,rounding_size=0.16",
                         linewidth=1.6, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
            fontsize=fontsize, linespacing=1.5, zorder=3, color="#111111")


def draw_edge(ax, n1, n2, color, lw=1.6, label=None, lpos=(0.5, 0.0, 0.0),
              label_fontsize=8.2, label_color="#333333"):
    (cx1, cy1, w1, h1) = n1
    (cx2, cy2, w2, h2) = n2
    x1, y1 = border_point(cx1, cy1, w1, h1, cx2, cy2)
    x2, y2 = border_point(cx2, cy2, w2, h2, cx1, cy1)
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=1))
    if label:
        t, dx, dy = lpos
        ax.text(x1 + (x2 - x1) * t + dx, y1 + (y2 - y1) * t + dy, label,
                ha="center", va="center", fontsize=label_fontsize, color=label_color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92), zorder=4)


def plot_network_graph(params, X, T, cache, L, grads, mids, outpath):
    """左：前向链（框里写张量名和形状）；右：反向链（框里写梯度名和形状 + 局部导数公式）。"""
    P = cache["P"]
    N = X.shape[0]
    g = lambda v: f"{v:.4g}"

    # 六个主链节点的布局（两张子图共用，方便左右对照）
    nX = (1.00, 3.60, 1.70, 1.00)
    nZ1 = (3.55, 3.60, 2.05, 1.00)
    nA1 = (6.05, 3.60, 1.95, 1.00)
    nZ2 = (8.55, 3.60, 2.05, 1.00)
    nP = (11.05, 3.60, 1.95, 1.00)
    nL = (13.45, 3.60, 1.65, 1.00)
    nW1 = (3.55, 1.60, 2.05, 1.05)
    nB1 = (3.55, 5.70, 2.05, 1.05)
    nW2 = (8.55, 1.60, 2.05, 1.05)
    nB2 = (8.55, 5.70, 2.05, 1.05)

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 6.6))
    for ax in axes:
        ax.set_xlim(-0.30, 15.05)
        ax.set_ylim(0.10, 6.95)
        ax.axis("off")

    # ---------------- 左：前向 ----------------
    ax = axes[0]
    ax.set_title(f"前向计算图：X → Linear → ReLU → Linear → Sigmoid → BCE"
                 f"（N = {N}，隐藏层 {params['W1'].shape[1]} 个神经元）\n"
                 f"框里是张量名和形状，边上是这一步的运算；loss = {g(L)}",
                 fontsize=11)
    draw_node(ax, *nX, ["X 输入", shp(X)], "#2f6f9f", "#e8f2fa")
    draw_node(ax, *nZ1, ["Z1 = X W1 + b1", shp(cache["Z1"])], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nA1, ["A1 = relu(Z1)", shp(cache["A1"])], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nZ2, ["Z2 = A1 W2 + b2", shp(cache["Z2"])], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nP, ["P = σ(Z2)", shp(P)], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nL, ["L = BCE", "标量", f"= {g(L)}"], "#a33b3b", "#fdecec")
    draw_node(ax, *nW1, ["W1  参数", shp(params["W1"])], "#6b4f9e", "#f1ecfb")
    draw_node(ax, *nB1, ["b1  参数", shp(params["b1"])], "#6b4f9e", "#f1ecfb")
    draw_node(ax, *nW2, ["W2  参数", shp(params["W2"])], "#6b4f9e", "#f1ecfb")
    draw_node(ax, *nB2, ["b2  参数", shp(params["b2"])], "#6b4f9e", "#f1ecfb")

    draw_edge(ax, nX, nZ1, "#4a7ea8", label="Linear", lpos=(0.5, 0.0, 0.34))
    draw_edge(ax, nZ1, nA1, "#4a7ea8", label="ReLU", lpos=(0.5, 0.0, 0.32))
    draw_edge(ax, nA1, nZ2, "#4a7ea8", label="Linear", lpos=(0.5, 0.0, 0.34))
    draw_edge(ax, nZ2, nP, "#4a7ea8", label="Sigmoid", lpos=(0.5, 0.0, 0.32))
    draw_edge(ax, nP, nL, "#4a7ea8", label="BCE + mean", lpos=(0.5, 0.0, 0.34))
    draw_edge(ax, nW1, nZ1, "#6b4f9e", label="X W1", lpos=(0.5, 0.30, 0.0))
    draw_edge(ax, nB1, nZ1, "#6b4f9e", label="+ b1（广播）", lpos=(0.5, 0.34, 0.0))
    draw_edge(ax, nW2, nZ2, "#6b4f9e", label="A1 W2", lpos=(0.5, 0.30, 0.0))
    draw_edge(ax, nB2, nZ2, "#6b4f9e", label="+ b2（广播）", lpos=(0.5, 0.34, 0.0))
    ax.text(7.5, 0.46, f"X 的形状 {shp(X)} 是 (样本数, 2)；每个样本 2 个特征进、1 个概率出。"
                       f"  b1/b2 是 (H,)/(1,) 却能加到 (N,H)/(N,1) 上，靠的是广播。",
            ha="center", fontsize=8.6, color="#555555")

    # ---------------- 右：反向 ----------------
    ax = axes[1]
    ax.set_title("反向计算图：从 loss 倒着走，每步乘上本层的局部导数\n"
                 "红箭头 = 梯度回传方向，边上是局部导数公式，框里是 ∂L/∂· 的形状",
                 fontsize=11)
    draw_node(ax, *nX, ["X 输入", f"∂L/∂X {shp(X)}", "（本题不需要）"], "#b3451f", "#ffe9dc")
    draw_node(ax, *nZ1, ["∂L/∂Z1", shp(mids["dZ1"])], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nA1, ["∂L/∂A1", shp(mids["dA1"])], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nZ2, ["∂L/∂Z2", shp(mids["dZ2"])], "#8c6d1f", "#fdf6e3")
    draw_node(ax, *nP, ["∂L/∂P", shp(P)], "#2e7d5b", "#e9f7f0")
    draw_node(ax, *nL, ["L（起点）", "∂L/∂L = 1"], "#a33b3b", "#fdecec")
    draw_node(ax, *nW1, ["∂L/∂W1", shp(grads["W1"]),
                         "= X^T · ∂L/∂Z1"], "#b3451f", "#ffe9dc")
    draw_node(ax, *nB1, ["∂L/∂b1", shp(grads["b1"]),
                         "= sum(∂L/∂Z1)"], "#b3451f", "#ffe9dc")
    draw_node(ax, *nW2, ["∂L/∂W2", shp(grads["W2"]),
                         "= A1^T · ∂L/∂Z2"], "#b3451f", "#ffe9dc")
    draw_node(ax, *nB2, ["∂L/∂b2", shp(grads["b2"]),
                         "= sum(∂L/∂Z2)"], "#b3451f", "#ffe9dc")

    draw_edge(ax, nL, nP, "#c0392b", label="∂L/∂P = -(1/N)·[T/P - (1-T)/(1-P)]",
              lpos=(0.5, 0.0, -0.62), label_fontsize=7.4)
    draw_edge(ax, nP, nZ2, "#c0392b", label="∂P/∂Z2 = P(1-P)",
              lpos=(0.5, 0.0, 1.15), label_fontsize=7.4)
    draw_edge(ax, nZ2, nA1, "#c0392b", label="∂Z2/∂A1 = W2^T", lpos=(0.5, 0.0, 0.36))
    draw_edge(ax, nA1, nZ1, "#c0392b", label="∂A1/∂Z1 = 1{Z1>0}", lpos=(0.5, 0.0, 0.44))
    draw_edge(ax, nZ2, nW2, "#c0392b", label="∂Z2/∂W2 = A1^T", lpos=(0.5, 0.30, 0.0))
    draw_edge(ax, nZ2, nB2, "#c0392b", label="∂Z2/∂b2 = 1", lpos=(0.5, 0.30, 0.0))
    draw_edge(ax, nZ1, nW1, "#c0392b", label="∂Z1/∂W1 = X^T", lpos=(0.5, 0.30, 0.0))
    draw_edge(ax, nZ1, nB1, "#c0392b", label="∂Z1/∂b1 = 1", lpos=(0.5, 0.30, 0.0))
    ax.text(7.5, 0.46, "两条最重要的“形状规则”：① 参数梯度和参数本身形状相同；"
                       "② 每一层的梯度形状和该层前向输出相同。\n"
                       "③ ∂P/∂Z2 = P(1-P) 与 BCE 导数里的 P(1-P) 相消 → ∂L/∂Z2 = (P - T)/N"
                       "（框架里的 BCEWithLogitsLoss 用的就是这一步）",
            ha="center", fontsize=8.6, color="#8e2b1c")

    fig.suptitle("实践 2：两层网络 X → Linear → ReLU → Linear → Sigmoid → loss 的计算图", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"已保存计算图：{outpath}")


def wrap_cjk(text, width=98):
    """按“显示宽度”折行：中文/全角字符算 2，ASCII 算 1，避免文字框超出画布。"""
    out = []
    for para in text.split("\n"):
        line, w = "", 0
        for ch in para:
            cw = 2 if ord(ch) > 0x2E80 else 1
            if w + cw > width:
                out.append(line)
                line, w = ch, cw
            else:
                line += ch
                w += cw
        out.append(line)
    return "\n".join(out)


def plot_grad_check(params, grads_np, grads_torch, grads_num, mids, err_torch,
                    err_num, bug_rows, outpath):
    """三面板：逐项梯度对比 / 逐项绝对误差 / 典型错误诊断。"""
    names, np_vals, th_vals, num_vals, errors = [], [], [], [], []
    for k in ("W1", "b1", "W2", "b2"):
        a, b, n, c = grads_np[k], grads_torch[k], grads_num[k], None
        for idx in np.ndindex(a.shape):
            names.append(f"{k}[{','.join(str(i) for i in idx)}]")
            np_vals.append(float(a[idx]))
            num_vals.append(float(n[idx]))
            th_vals.append(float(b[idx]))
            errors.append(abs(float(a[idx] - b[idx])))
    np_vals, th_vals, num_vals, errors = map(np.array, (np_vals, th_vals, num_vals, errors))

    fig = plt.figure(figsize=(16.4, 9.8))
    ax1 = fig.add_axes((0.05, 0.42, 0.42, 0.47))
    ax2 = fig.add_axes((0.545, 0.42, 0.42, 0.47))
    axt = fig.add_axes((0.035, 0.02, 0.94, 0.30))
    axt.axis("off")

    # ---- 左：13 个参数逐项对比（三种来源）----
    pos = np.arange(len(names), dtype=float)
    w = 0.27
    ax1.bar(pos - w, np_vals, w, label="NumPy 手写反向", color="#2f6f9f")
    ax1.bar(pos, num_vals, w, label="中心差分数值梯度", color="#7f8c8d")
    ax1.bar(pos + w, th_vals, w, label="PyTorch autograd", color="#c0392b", alpha=0.85)
    ax1.axhline(0.0, color="#888888", lw=0.9)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(names, rotation=70, fontsize=7.6, ha="right")
    ax1.set_ylabel("梯度数值", fontsize=9.5)
    ax1.set_title(f"13 个参数的梯度逐项对比（三根柱子应当齐平）\n"
                  f"手写 vs autograd 最大绝对误差 {err_torch:.2e} ；"
                  f"手写 vs 中心差分最大绝对误差 {err_num:.2e}", fontsize=10)
    ax1.legend(fontsize=8.4, loc="best")
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.set_xlim(-0.8, len(names) - 0.2)

    # ---- 右：误差量级 + 诊断信息 ----
    ax2.set_yscale("log")
    eps_floor = 1e-18
    ax2.bar(pos, np.maximum(errors, eps_floor), 0.55, color="#8e44ad", alpha=0.85,
            label="|手写 - autograd| 逐项绝对误差")
    ax2.axhline(1e-12, color="#c0392b", ls="--", lw=1.2, label="1e-12（视为一致的门槛）")
    ax2.axhline(np.maximum(err_num, eps_floor), color="#7f8c8d", ls=":", lw=1.4,
                label=f"数值梯度的误差水平 {err_num:.1e}")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(names, rotation=70, fontsize=7.6, ha="right")
    ax2.set_ylabel("绝对误差（对数轴）", fontsize=9.5)
    ax2.set_ylim(eps_floor, 1e-2)
    ax2.set_title("误差有多大、以及“算错了会长什么样”", fontsize=10)
    ax2.legend(fontsize=8.2, loc="upper left")
    ax2.grid(axis="y", alpha=0.3, linestyle=":", which="both")
    ax2.set_xlim(-0.8, len(names) - 0.2)

    txt = ("诊断经验（数值对不上时按这个顺序查）：\n"
           "  1) 形状对不上 → 多半是转置/维度顺序，先把每一步的形状 print 出来\n"
           "  2) 形状对、但所有项都差同一个固定倍数 → 求平均方式（sum vs mean）\n"
           "  3) 只有一部分元素差 → 某个逐元素操作写错了（ReLU 掩码位置、广播方向）\n"
           "  4) 只差 1e-5 ~ 1e-8 → 多半是精度：dtype 用了 float32，或数值微分 h 没选好")
    axt.text(0.0, 1.0, wrap_cjk(txt), transform=axt.transAxes, fontsize=9.0,
             va="top", ha="left", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf5",
                       edgecolor="#d8cfa8", alpha=0.97))
    if bug_rows:
        lines = ["本脚本故意制造的错误（用来认识症状）："]
        for name, detail in bug_rows:
            lines.append(f"  · {name}：{detail}")
        axt.text(0.0, 0.42, wrap_cjk("\n".join(lines)), transform=axt.transAxes,
                 fontsize=8.8, va="top", ha="left", linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="#fdf0f0",
                           edgecolor="#e0b4b4", alpha=0.97))

    fig.suptitle("实践 2：NumPy 手写梯度 vs 中心差分 vs PyTorch autograd", fontsize=13.5, y=0.97)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"已保存验证图：{outpath}")


def plot_batch_effect(params, res_single, res_batch, sum_ratio, outpath):
    """单样本 vs 小 batch：形状怎么变、1/N 因子从哪来、sum/mean 混淆的症状。"""
    X1, cache1, grads1 = res_single["X"], res_single["cache"], res_single["grads_np"]
    Xb, cacheb, gradsb = res_batch["X"], res_batch["cache"], res_batch["grads_np"]

    fig = plt.figure(figsize=(16.0, 6.0))
    ax1 = fig.add_axes((0.05, 0.14, 0.42, 0.70))
    ax2 = fig.add_axes((0.55, 0.14, 0.42, 0.70))

    # ---- 左：形状对照表 ----
    rows = [
        ("X", shp(X1), shp(Xb)),
        ("Z1 = X W1 + b1", shp(cache1["Z1"]), shp(cacheb["Z1"])),
        ("A1 = relu(Z1)", shp(cache1["A1"]), shp(cacheb["A1"])),
        ("Z2 = A1 W2 + b2", shp(cache1["Z2"]), shp(cacheb["Z2"])),
        ("P = σ(Z2)", shp(cache1["P"]), shp(cacheb["P"])),
        ("W1 / ∂L/∂W1", shp(params["W1"]), shp(params["W1"])),
        ("b1 / ∂L/∂b1", shp(params["b1"]), shp(params["b1"])),
        ("W2 / ∂L/∂W2", shp(params["W2"]), shp(params["W2"])),
        ("b2 / ∂L/∂b2", shp(params["b2"]), shp(params["b2"])),
    ]
    ax1.axis("off")
    ax1.set_title("换成一个 batch 之后：变的是“带样本数的那一维”\n"
                  "参数的形状和参数梯度的形状完全不变", fontsize=10.5)
    col_x = (0.02, 0.48, 0.71)
    ax1.text(col_x[1], 0.955, "N = 1（单个样本）", fontsize=9.6, weight="bold", ha="left")
    ax1.text(col_x[2], 0.955, "N = 8（小 batch）", fontsize=9.6, weight="bold", ha="left")
    for r, (name, s1, sb) in enumerate(rows):
        yy = 0.895 - r * 0.088
        ax1.text(col_x[0], yy, name, fontsize=9.0, ha="left")
        ax1.text(col_x[1], yy, s1, fontsize=9.0, ha="left", color="#1f6f43")
        ax1.text(col_x[2], yy, sb, fontsize=9.0, ha="left", color="#b3451f" if s1 != sb else "#1f6f43")
    ax1.text(0.02, 0.075,
             "公式上真正变化的地方只有三处：\n"
             "  ① loss 从“单个样本的损失”变成 mean(各样本损失) → 反向多出 1/N 因子\n"
             "  ② ∂L/∂b = 要对 batch 维求和（sum over axis=0），否则形状对不上\n"
             "  ③ ∂L/∂W = X^T @ dZ 里的 X^T / dZ 多了一维，矩阵乘自动把 batch 维“吃掉”\n"
             "  而 ReLU 掩码、Sigmoid 导数、dZ2 = (P-T)/N 这些逐元素的公式一个字都不用改。",
             fontsize=8.6, ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f9fb",
                       edgecolor="#c8d4de", alpha=0.97))

    # ---- 右：sum vs mean 的症状 ----
    ax2.set_title("把 mean 写成 sum 会长什么样：所有梯度被同一个倍数放大", fontsize=10.5)
    names = [f"{k}[{','.join(str(i) for i in idx)}]"
             for k in ("W1", "b1", "W2", "b2") for idx in np.ndindex(gradsb[k].shape)]
    ratios = np.array(sum_ratio, dtype=float)
    n_bug = float(np.median(ratios))
    ax2.bar(np.arange(len(names)), ratios, 0.55, color="#c0392b", alpha=0.6,
            label=f"sum 版 / mean 版（每一项都 = {n_bug:g}）")
    ax2.axhline(1.0, color="#2f6f9f", lw=1.8, ls="--", label="正确（mean）时的比值 = 1")
    ax2.text(0.02, 0.80, f"对每个参数算 “sum 版梯度 ÷ mean 版梯度”，\n"
                         f"所有 {len(names)} 项都等于同一个数 {n_bug:.6f} = N，\n"
                         f"形状还全都正确 —— 这就是“求平均方式错了”的指纹。\n"
                         f"（反过来：如果只是某几项倍数不对，那是转置或广播错了。）",
             transform=ax2.transAxes, fontsize=8.8, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf5",
                       edgecolor="#d8cfa8", alpha=0.97))
    ax2.set_xlabel("b1 / W2 / b2 等 13 个参数元素（按顺序）", fontsize=9.5)
    ax2.set_ylabel("sum 版梯度 / mean 版梯度", fontsize=9.5)
    ax2.set_ylim(0, max(float(np.max(ratios)) * 1.25, 2.0))
    ax2.legend(fontsize=8.4, loc="center right")
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.set_xticks([])

    fig.suptitle("实践 2：单个样本 → 小 batch，哪些公式和形状变了", fontsize=13.5, y=0.96)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"已保存 batch 对照图：{outpath}")


# ==================================================================
# 九、一次完整的“前向 + 反向 + 三方对比”，单样本 / batch 共用同一套代码
# ==================================================================
def run_once(params, X, T, label, verbose=True):
    if verbose:
        print_params(params)
    L, cache = forward_numpy(params, X, T)
    grads_np, mids = backward_numpy(params, cache, fused=True)
    _, mids_full = backward_numpy(params, cache, fused=False)
    L_torch, grads_torch, mids_torch = torch_forward_backward(params, X, T)
    grads_num = numeric_grad(params, X, T)
    rows, err_torch = compare(grads_np, grads_torch)
    err_num = max(float(np.max(np.abs(grads_np[k] - grads_num[k]))) for k in grads_np)
    return {"label": label, "X": X, "T": T, "L": L, "L_torch": L_torch, "cache": cache,
            "grads_np": grads_np, "grads_torch": grads_torch, "grads_num": grads_num,
            "mids": mids, "mids_full": mids_full, "mids_torch": mids_torch,
            "rows": rows, "err_torch": err_torch, "err_num": err_num}


def print_comparison(res):
    print(f"\n  逐项比较（{res['label']}）：")
    print(f"    {'项':>4} {'形状':>9} {'NumPy 手写':>16} {'PyTorch autograd':>18} "
          f"{'中心差分':>16} {'最大绝对误差':>12}  一致？")
    for k, shape, a, b, e in res["rows"]:
        num = res["grads_num"][k]
        print(f"    {k:>4} {shp(shape):>9} {np.array2string(a.reshape(-1), precision=8, suppress_small=True):>16} "
              f"{np.array2string(b.reshape(-1), precision=8, suppress_small=True):>18} "
              f"{np.array2string(num.reshape(-1), precision=8, suppress_small=True):>16} "
              f"{e:>12.2e}  {'✔' if e < 1e-12 else '✘'}")
    print(f"    loss：手写 {res['L']:.12f} , PyTorch {res['L_torch']:.12f} , "
          f"差 {abs(res['L'] - res['L_torch']):.2e}  "
          f"{'✔' if abs(res['L'] - res['L_torch']) < 1e-12 else '✘'}")
    print(f"    中间量：Z1 最大差 {float(np.max(np.abs(res['cache']['Z1'] - res['mids_torch']['Z1']))):.2e} , "
          f"A1 最大差 {float(np.max(np.abs(res['cache']['A1'] - res['mids_torch']['A1']))):.2e} , "
          f"Z2 最大差 {float(np.max(np.abs(res['cache']['Z2'] - res['mids_torch']['Z2']))):.2e} , "
          f"P 最大差 {float(np.max(np.abs(res['cache']['P'] - res['mids_torch']['P']))):.2e}")
    print(f"    手写 vs autograd 最大绝对误差 = {res['err_torch']:.2e} ；"
          f"手写 vs 中心差分 = {res['err_num']:.2e}")
    print(f"    结论：{'完全一致（误差在浮点舍入量级，逐项都过 1e-12 门槛）' if res['err_torch'] < 1e-12 else '不一致，需要检查公式'}")


# ==================================================================
# 十、故意写错的三种版本：认识“症状”
# ==================================================================
def bug_patterns(params, X, T, res):
    """三种典型错误各跑一次，记录症状，用于“出问题怎么查”那一问。"""
    rows = []
    N = X.shape[0]
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    ref = res["grads_torch"]

    # 错误 1：loss 用 sum 而不是 mean（把 N 个样本的损失加起来）
    _, cache_sum = forward_numpy(params, X, T)
    dZ2_sum = (cache_sum["P"] - cache_sum["T"])            # 少了 /N
    dA1_sum = dZ2_sum @ params["W2"].T
    dZ1_sum = dA1_sum * (cache_sum["Z1"] > 0.0)
    grads_sum = {"W1": cache_sum["X"].T @ dZ1_sum, "b1": dZ1_sum.sum(axis=0),
                 "W2": cache_sum["A1"].T @ dZ2_sum, "b2": dZ2_sum.sum(axis=0)}
    e_sum = float(np.max(np.abs(grads_sum["W2"] - ref["W2"])))
    ratio_list = []
    for k in ("W1", "b1", "W2", "b2"):
        ratio_list.extend((grads_sum[k] / ref[k]).reshape(-1).tolist())
    ratios = np.array(ratio_list)
    ratio_gap = float(np.max(np.abs(ratios - N)))
    rows.append(("★ 把 mean 写成 sum（loss 没除以 N）",
                 f"每个梯度都被放大恰好 N={N} 倍（比值偏离 N 的最大量 "
                 f"{ratio_gap:.1e}），形状却完全正确"))
    print("\n  【故意写错 1】loss 用 sum 而不是 mean：")
    print(f"    13 项梯度与正确值的比值（全部应当 = {N}）："
          f"{np.array2string(ratios, precision=6)}")
    print(f"    最大绝对误差 {e_sum:.4e}（正确版本是 {res['err_torch']:.2e}）")
    print("    → 症状：所有参数、所有元素都差同一个倍数；形状没错。"
          " 这几乎一定是求平均/求和的问题。")

    # 错误 2：转置写反 —— dW1 = dZ1.T @ X 之类
    dZ1 = res["mids"]["dZ1"]
    wrong_shape = (dZ1.T @ X)                  # (H,N)@(N,2) → (H,2)，和 W1 (2,H) 形状不同
    rows.append(("★ 转置写反（dW1 写成 dZ1^T @ X）",
                 f"形状直接变成 {shp(wrong_shape)}，而 W1 是 {shp(W1)} → 报错/必须强行 reshape"))
    print("\n  【故意写错 2】把 dW1 = Xᵀ @ dZ1 写成 dW1 = dZ1ᵀ @ X：")
    print(f"    形状变成 {shp(wrong_shape)}，而 W1 的形状是 {shp(W1)}，"
          f"一相减就会抛 broadcast 错误。")
    # 形状凑巧相同但数值错的版本：dW1 = Xᵀ @ dZ1 但把 X 换成 A1（张冠李戴）
    dW1_silly = res["cache"]["A1"].T @ dZ1      # (H,N)@(N,H) → (H,H)，这里 H=2 时才行
    if dW1_silly.shape == W1.shape:
        e_silly = float(np.max(np.abs(dW1_silly - ref["W1"])))
        rows.append(("★ 张冠李戴（dW1 用 A1^T 而不是 X^T）",
                     f"形状侥幸对得上，但数值最大绝对误差 {e_silly:.3e}，只有部分元素错"))
        print(f"    若上面写成 A1^T @ dZ1（隐藏层恰好也是 2 维时形状都相同）："
              f"最大绝对误差 {e_silly:.3e}，形状看不出来，只能靠数值对比发现。")

    # 错误 3：b 的梯度忘记对 batch 求和
    db1_rowsum = res["mids"]["dZ1"]            # (N,H)，没有 sum(axis=0)
    rows.append(("★ 忘记对 batch 求和（db1 少了 sum(axis=0)）",
                 f"形状是 {shp(db1_rowsum)}，而 b1 是 {shp(b1)}：" +
                 ("N=1 时看不出问题，N>1 时形状对不上" if N > 1 else "单样本时恰好不暴露")))
    print("\n  【故意写错 3】b1 的梯度忘了对 batch 维求和：")
    print(f"    拿到的是 {shp(db1_rowsum)}（每个样本一份），b1 的形状是 {shp(b1)}。")
    if N > 1:
        print("    → 形状对不上；而且 numpy 的广播还会“悄悄帮你加”，"
              "最后得到 (N,H) 的梯度，用它更新 b1 时行为就错了。")
        print(f"    → 症状：只有偏置相关的那几项错，且错得和 batch 大小有关，"
              f"db1 逐行之和 = {np.array2string(db1_rowsum.sum(axis=0), precision=6)} 才是正确值。")
    else:
        print("    → 单个样本时 batch 维长度是 1，求不求和结果一样，所以这个 bug 只在 batch 里出现。")
    return rows, ratios


# ==================================================================
# 十一、主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="实践 2：两层网络的梯度（NumPy 手写 vs autograd）")
    ap.add_argument("--n-hidden", type=int, default=3, help="隐藏层神经元个数（默认 3）")
    ap.add_argument("--batch-size", type=int, default=8, help="小 batch 的样本数（默认 8）")
    ap.add_argument("--sample-index", type=int, default=0, help="取第几个训练样本（默认 0）")
    ap.add_argument("--outdir", type=str, default=HERE, help="图片输出目录")
    ap.add_argument("--no-figures", action="store_true", help="只打印，不画图")
    args = ap.parse_args()
    outdir = _os.path.abspath(args.outdir)
    _os.makedirs(outdir, exist_ok=True)

    section("任务 0：数据（Task 1 的 make_moons）与网络结构")
    (X1, T1), (Xb, Tb), (Xtr, ytr) = load_single_and_batch(args.sample_index, args.batch_size)
    print(f"  make_moons(n_samples=1000, noise=0.2, random_state={SEED}) → "
          f"train/val/test 划分 → StandardScaler（只在训练集 fit），与 Task 1 完全一致")
    print(f"  训练集样本数 {len(Xtr)}；本次取第 {args.sample_index} 个训练样本作为“一个样本”的例子：")
    print(f"    原始（标准化后）特征 X = {np.array2string(X1[0], precision=6)} , 标签 t = {int(T1[0, 0])}")
    print(f"    小 batch：前 {args.batch_size} 个训练样本，X 形状 {shp(Xb)}，标签 T 形状 {shp(Tb)}")
    print(f"  网络：X → Linear → ReLU → Linear → Sigmoid → BCE，隐藏层 {args.n_hidden} 个神经元")
    print(f"  参数量：2×{args.n_hidden} + {args.n_hidden} + {args.n_hidden}×1 + 1 = "
          f"{2 * args.n_hidden + args.n_hidden + args.n_hidden + 1} 个")
    print("  形状约定：W1 是 (输入维, 隐藏维) = (2, H)，W2 是 (H, 1)——")
    print("  因为手写时用 X @ W1；PyTorch 的 nn.Linear 内部是 x @ W.T，它存的是 (out, in) = (H, 2)。")

    section("任务 1：自定义隐藏层大小并初始化 W1 / b1 / W2 / b2")
    params = init_params(n_in=2, n_hidden=args.n_hidden, n_out=1, seed=0)
    print_params(params)
    print("  初始化用标准正态 N(0, sqrt(2/fan_in))，偏置全 0；"
          "隐藏层取这么小是为了 13 个梯度能一个个盯住。")

    section("任务 2 + 3：NumPy 前向 + 反向（一个样本）")
    print("  --- 前向 ---")
    L1, cache1 = forward_numpy(params, X1, T1)
    print_forward_chain(params, X1, T1, cache1, L1)
    print("\n  --- 反向（第一个样本只有 1/N = 1，所以和 batch 的差别只在“求和 vs 求平均”）---")
    grads1, mids1 = backward_numpy(params, cache1, fused=True)
    print_backward_chain(params, cache1, grads1, mids1)
    print(f"\n  结果：loss = {L1:.10f}，P = {float(cache1['P'][0, 0]):.10f}，标签 t = {int(T1[0, 0])}"
          f"（预测概率{'高于' if cache1['P'][0, 0] > 0.5 else '低于'} 0.5）")
    for k in ("W1", "b1", "W2", "b2"):
        print(f"    ∂L/∂{k:3s} = {np.array2string(grads1[k], precision=8, suppress_small=True)}")

    section("任务 4 + 5：用 PyTorch autograd 复核（同一个样本、同一组参数）")
    res1 = run_once(params, X1, T1, "N = 1（单个样本）", verbose=False)
    print_comparison(res1)
    print("\n  autograd 自己建的反向图（也就是它认为的计算图）：")
    for line in torch_graph_lines(params, X1, T1):
        print("   " + line)
    print("  这张图和上面画的计算图逐层对应：每个 MmBackward0（矩阵乘的反向节点）对应一次 Linear，")
    print("  ReluBackward0 对应 ReLU，SigmoidBackward0 + BinaryCrossEntropyBackward0 对应最后两步，")
    print("  而每个参数下挂的 AccumulateGrad 就是“梯度累加到 .grad”这件事。")

    section(f"任务 5（加强）：换成一个小 batch（N = {args.batch_size}）再看一遍")
    print("  --- 前向 ---")
    Lb, cacheb = forward_numpy(params, Xb, Tb)
    print_forward_chain(params, Xb, Tb, cacheb, Lb)
    print("\n  --- 反向：只有三处和单样本不同（1/N、对 batch 求和、矩阵乘多一维）---")
    gradsb, midsb = backward_numpy(params, cacheb, fused=True)
    print_backward_chain(params, cacheb, gradsb, midsb)
    resb = run_once(params, Xb, Tb, f"N = {args.batch_size}（小 batch）", verbose=False)
    print_comparison(resb)
    print("\n  哪些东西变了：")
    print("    · 变化 1：loss 从“单个样本的损失”变成“batch 平均”，反向多带一个 1/N 因子；")
    print("              dZ2 = (P - T)/N 里的 /N 就是为了它。")
    print("    · 变化 2：∂L/∂b 多了一步 sum(axis=0)——把每个样本的贡献加起来；")
    print("              偏置在 batch 里所有样本间是共享的，所以它的梯度要把 batch 维消掉。")
    print("    · 变化 3：∂L/∂W 的矩阵乘从 (1,N)@(N,H) 变成 (2,N)@(N,H)，")
    print("              batch 维在矩阵乘里被自动求和消掉——这正是“样本共享权重”的表达。")
    print("    · 没变的：ReLU 掩码、Sigmoid 导数、dZ2 = (P-T)/N 的逐元素形式、")
    print("              参数及其梯度的形状（(2,H)、(H,)、(H,1)、(1,)）。")

    section("附加：三个“故意写错”的版本，认一认症状")
    bug_rows, sum_ratios = bug_patterns(params, Xb, Tb, resb)

    if not args.no_figures:
        section("出图")
        plot_network_graph(params, Xb, Tb, cacheb, Lb, gradsb, midsb,
                           _os.path.join(outdir, "03_nn_compute_graph.png"))
        plot_grad_check(params, resb["grads_np"], resb["grads_torch"], resb["grads_num"],
                        midsb, resb["err_torch"], resb["err_num"], bug_rows,
                        _os.path.join(outdir, "04_nn_grad_check.png"))
        plot_batch_effect(params, res1, resb, sum_ratios,
                          _os.path.join(outdir, "05_nn_batch_effect.png"))

    section("总结")
    print(f"  1) 前向：Z1({shp(cacheb['Z1'])}) → A1({shp(cacheb['A1'])}) → Z2({shp(cacheb['Z2'])}) → "
          f"P({shp(cacheb['P'])}) → loss（标量 {Lb:.10f}）")
    print(f"  2) 反向：dZ2 = (P-T)/N → dW2 = A1ᵀ·dZ2、db2 = ΣdZ2 → dA1 = dZ2·W2ᵀ → "
          f"dZ1 = dA1⊙1{{Z1>0}} → dW1 = Xᵀ·dZ1、db1 = ΣdZ1")
    print(f"  3) 手写 NumPy 与 PyTorch autograd 的最大绝对误差 = {resb['err_torch']:.2e}，"
          f"与中心差分 = {resb['err_num']:.2e} → 公式正确。")
    print("  4) 单个样本和 batch 的差别只有三处：1/N、对 batch 求和、矩阵乘多一维；")
    print("     参数形状与参数梯度形状永远不变，这是检查代码时最先看的东西。")


if __name__ == "__main__":
    main()
