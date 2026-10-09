# -*- coding: utf-8 -*-
"""
Jotang ML 项目：moons 数据集上的 MLP 二分类

任务 1：可视化原始数据；划分 训练/验证/测试集；说明三者职责、避免数据泄漏
任务 2：用 torch 搭建「至少包含一个隐藏层」的 MLP，完成 训练/验证/测试/模型保存与加载
任务 3：绘制 loss、accuracy 曲线 及 二维决策边界
任务 4：对照实验（一次只改一个主要变量），观察指标与结果变化
任务 5：输出混淆矩阵，并挑几个判错的样本做分析

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python MLP.py
    2) conda run -n jotang-ml python MLP.py
    3) 在 VSCode / Jupyter 里直接运行（已内置“环境自举”，见第 0 步）
"""

# ==================================================================
# 【第 0 步】运行环境自举（关键修复，不要删）
# ------------------------------------------------------------------
# 现象：不激活 conda 环境而直接运行 python.exe 时，进程在做矩阵乘
#       （np.dot / @）的一瞬间会原生崩溃，报
#       "Windows fatal exception: code 0xc06d007f"，而且没有任何 Python 报错。
# 原因：本环境 numpy 用的 MKL 放在 conda 环境的 Library\bin 下；
#       不激活环境时该目录不在 PATH 上，numpy 会载入错误的 BLAS。
#       而 matplotlib 的 scatter / savefig 内部要用 np.dot，
#       所以“画图”这一步会跟着崩，看起来就像整个脚本跑不了。
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

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 是否弹出交互式图像窗口。True 会阻塞程序直到你关闭窗口，所以默认 False；
# 所有图都已经保存成 png 文件了。
SHOW_PLOTS = False

# 让图里的中文正常显示（Windows 自带雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================================================================
# 任务 1：可视化原始数据 + 划分 Train / Val / Test + 避免数据泄漏
# ------------------------------------------------------------------
# 【三个集合各自的职责】
#   · 训练集 Train：模型真正“学习”用的数据。每个 epoch 用它前向计算、
#     算 loss、反向传播并更新权重；它是唯一直接参与参数优化的集合。
#   · 验证集 Val  ：训练过程中的“模拟考”。**不参与梯度更新**，只用来
#     观察模型在没见过的数据上表现如何，并据此挑超参数 / 选最佳 epoch
#     （本脚本就是按验证集准确率保存 best_model.pth）。
#   · 测试集 Test ：最终“期末考”。等训练和调参全部结束之后只用一次，
#     用来估计模型对真实未知数据的泛化能力。若反复拿它调参，
#     它就会变相变成验证集，最后报出的成绩会偏乐观、不可信。
#
# 【怎么避免数据泄漏（data leakage）】
#   1) 先划分，再做任何“会从数据里学参数”的预处理。本脚本把
#      train_test_split 放在最前面，之后才做标准化。
#   2) StandardScaler **只在训练集上 fit**（学 mean/std），再用同一组
#      mean/std 去 transform 验证集和测试集。如果对全体数据一起 fit，
#      训练阶段就间接“看到”了验证/测试集的统计信息，这就是数据泄漏。
#      所以下面会看到：验证集/测试集标准化后的 mean 不是严格的 0、
#      std 不是严格的 1，这是正常的、也是正确的。
#   3) 调参、选模型只在 训练集 + 验证集 范围内进行；测试集从头到尾
#      不参与任何决策，只在最后评估一次。
#   4) 划分时用 stratify=y 让每个集合的类别比例保持一致，只是为了让
#      评估更稳定，它不影响上面这些原则。
# ==================================================================
X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)
print(f"原始数据: X.shape = {X.shape}, y.shape = {y.shape}")
print(f"类别分布: 0 有 {(y == 0).sum()} 个, 1 有 {(y == 1).sum()} 个")

# ---------- 1.1 可视化原始数据（选择你喜欢的方式）----------
plt.figure(figsize=(10, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')
plt.title("Moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("moons_dataset.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close("all")

# ---------- 1.2 按 训练 / 验证 / 测试 划分 ----------
# 先切出 20% 作测试集，再从剩下的里切出验证集，保证测试集从头到尾没被“看过”。
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=SEED, stratify=y_train_val)

print("\n划分后:")
print(f"  Train: {X_train.shape}, 标签比例 {np.bincount(y_train) / len(y_train)}")
print(f"  Val  : {X_val.shape},   标签比例 {np.bincount(y_val)   / len(y_val)}")
print(f"  Test : {X_test.shape},  标签比例 {np.bincount(y_test)  / len(y_test)}")

# ---------- 1.3 标准化：只在训练集上 fit，再 transform 其他集合 ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # 只在训练集上学习 mean 和 std
X_val   = scaler.transform(X_val)         # 用训练集的 mean/std 去变换验证集
X_test  = scaler.transform(X_test)        # 同理，变换测试集

print(f"\n标准化后 (训练集): mean = {X_train.mean(axis=0)}, std = {X_train.std(axis=0)}")
print(f"标准化后 (验证集): mean = {X_val.mean(axis=0)},   std = {X_val.std(axis=0)}")
# 注意：验证集/测试集的 mean 不一定精确等于 0，std 不一定等于 1，这是正常的！
# 因为它们是用训练集的参数来变换的，这正是“避免数据泄漏”的体现。

# ---------- 1.4 可视化划分结果 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (Xs, ys, name, color) in zip(
    axes,
    [
        (X_train, y_train, f"Train ({len(y_train) / len(y) * 100:.0f}%)", 'tab:green'),
        (X_val,   y_val,   f"Val ({len(y_val)   / len(y) * 100:.0f}%)",   'tab:purple'),
        (X_test,  y_test,  f"Test ({len(y_test)  / len(y) * 100:.0f}%)",  'tab:red'),
    ]
):
    ax.scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], c=color, marker='o', s=20, alpha=0.7, label='Class 0')
    ax.scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], c=color, marker='^', s=20, alpha=0.7, label='Class 1')
    ax.set_title(name)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("split_data.png", dpi=150)
if SHOW_PLOTS:
    plt.show()
plt.close("all")

# ---------- 1.5 转成 PyTorch Tensor / DataLoader ----------
def make_loaders(Xtr, ytr, Xva, yva, Xte, yte, batch_size=64, shuffle_train=True):
    """把 numpy 数据包装成 (train_loader, val_loader, test_loader)。"""
    def to_x(a):
        return torch.tensor(a, dtype=torch.float32)
    def to_y(a):
        return torch.tensor(a, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(to_x(Xtr), to_y(ytr)),
                              batch_size=batch_size, shuffle=shuffle_train)
    val_loader   = DataLoader(TensorDataset(to_x(Xva), to_y(yva)), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(to_x(Xte), to_y(yte)), batch_size=batch_size)
    return train_loader, val_loader, test_loader

BATCH_SIZE = 64
train_loader, val_loader, test_loader = make_loaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE)
print(f"\nTensor 形状: X_train_t = ({len(y_train)}, 2), y_train_t = ({len(y_train)},)")

# ==================================================================
# 任务 2-①：搭建 MLP（至少包含一个隐藏层）
# ==================================================================
class MLP(nn.Module):
    """全连接前馈网络，结构可调。

    结构：2 -> [hidden_width] * n_hidden -> 2
    - 中间 n_hidden 层就是隐藏层（要求“至少一个”，默认给两层）；
    - 隐藏层激活函数可选 relu / tanh / sigmoid；
    - 输出层不接激活函数，直接输出 2 个 logits，交给 CrossEntropyLoss。
    """
    def __init__(self, hidden_width=32, n_hidden=2, activation='relu'):
        super().__init__()
        act = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[activation]
        layers, in_dim = [], 2
        for _ in range(n_hidden):
            layers += [nn.Linear(in_dim, hidden_width), act()]
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, 2))      # 输出层：2 个 logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==================================================================
# 任务 2-②：训练 + 验证
# ==================================================================
def train_and_validate(model, train_loader, val_loader, epochs=100, lr=1e-2,
                       opt_name="adam", save_path="best_model.pth", verbose=True):
    """在训练集上训练，每个 epoch 结束后在验证集上评估，
    并按验证集准确率保留“最佳模型”（save_path 为 None 时只留在内存里，不写文件）。
    返回 (训练历史, 最佳 epoch, 最佳验证准确率, 最佳权重)。"""
    criterion = nn.CrossEntropyLoss()
    if opt_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None
    start_time = time.time()

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        # ---- 验证 ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ---- 保存最佳模型（按验证集准确率）----
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(model.state_dict(), save_path)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n【训练完成】用时 {elapsed:.2f}s，"
              f"最佳验证准确率 {best_val_acc:.4f}（第 {best_epoch} 个 epoch）")
    return history, best_epoch, best_val_acc, best_state

@torch.no_grad()
def evaluate(model, loader, criterion):
    """返回给定数据集上的 (loss, accuracy)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def predict(model, loader_or_tensor):
    """返回 (预测类别, 两类概率)，输入可以是 DataLoader 或 Tensor。"""
    model.eval()
    if isinstance(loader_or_tensor, torch.Tensor):
        logits = model(loader_or_tensor.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba.argmax(1), proba
    preds, probas = [], []
    for xb, _ in loader_or_tensor:
        logits = model(xb.to(device))
        probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    proba = np.concatenate(probas, axis=0)
    return proba.argmax(1), proba

# ---------- 开始训练主模型 ----------
model = MLP(hidden_width=32, n_hidden=2, activation='relu').to(device)
history, best_epoch, best_val_acc, best_state = train_and_validate(
    model, train_loader, val_loader, epochs=100, lr=1e-2, save_path="best_model.pth")

print(f"\n【模型保存】最佳模型已保存到 best_model.pth"
      f"（第 {best_epoch} 个 epoch，验证准确率 {best_val_acc:.4f}）")

# ==================================================================
# 任务 2-③：模型加载 + 测试集评估
# ==================================================================
# 新建一个同结构的新模型，把保存的权重加载进去，
# 这样才能说明“保存 / 加载”是真的可用的。
loaded_model = MLP(hidden_width=32, n_hidden=2, activation='relu').to(device)
loaded_model.load_state_dict(torch.load("best_model.pth"))
print("\n【模型加载】已从 best_model.pth 重新载入权重")

criterion = nn.CrossEntropyLoss()
train_loss, train_acc = evaluate(loaded_model, train_loader, criterion)
val_loss,   val_acc   = evaluate(loaded_model, val_loader,   criterion)
test_loss,  test_acc  = evaluate(loaded_model, test_loader,  criterion)

print("【测试】加载后的模型评估结果：")
print(f"  Train: Loss {train_loss:.4f}, Acc {train_acc:.4f}")
print(f"  Val  : Loss {val_loss:.4f}, Acc {val_acc:.4f}")
print(f"  Test : Loss {test_loss:.4f}, Acc {test_acc:.4f}")

# ==================================================================
# 任务 3：绘制 loss / accuracy 曲线 及 二维决策边界
# ==================================================================
def plot_history(history, savepath="training_curves.png", title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title(f"{title} Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'],   label='Val Acc')
    axes[1].set_title(f"{title} Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(savepath, dpi=150); plt.close(fig)

def plot_decision_boundary(model, X_scaled, y_true, savepath,
                           title="决策边界（测试集）", mark_wrong=True):
    """在标准化后的二维特征平面上画出模型的分类区域与决策边界。"""
    model.eval()
    pad = 0.6
    x_min, x_max = X_scaled[:, 0].min() - pad, X_scaled[:, 0].max() + pad
    y_min, y_max = X_scaled[:, 1].min() - pad, X_scaled[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    zz = predict(model, grid)[0].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap='coolwarm', alpha=0.25)
    ax.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=1.2)
    ax.scatter(X_scaled[y_true == 0, 0], X_scaled[y_true == 0, 1],
               s=18, c='tab:blue', edgecolors='w', linewidths=0.5, label='真实 0')
    ax.scatter(X_scaled[y_true == 1, 0], X_scaled[y_true == 1, 1],
               s=18, c='tab:red', edgecolors='w', linewidths=0.5, label='真实 1')
    if mark_wrong:
        pred, _ = predict(model, torch.tensor(X_scaled, dtype=torch.float32))
        wrong = pred != y_true
        ax.scatter(X_scaled[wrong, 0], X_scaled[wrong, 1], s=110, facecolors='none',
                   edgecolors='k', linewidths=1.6, label=f'判错（{wrong.sum()} 个）')
    ax.set_title(title)
    ax.set_xlabel("Feature 1 (scaled)"); ax.set_ylabel("Feature 2 (scaled)")
    ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(savepath, dpi=150); plt.close(fig)

plot_history(history, "training_curves.png", "主模型")
plot_decision_boundary(loaded_model, X_test, y_test, "decision_boundary.png",
                       title="任务3：主模型在测试集上的决策边界")
print("\n【任务3】已保存 training_curves.png 与 decision_boundary.png")

# ==================================================================
# 任务 4：对照实验（一次只改变一个主要变量）
# ------------------------------------------------------------------
# 基线配置：width=32, 2 个隐藏层, ReLU, Adam, lr=1e-2, batch=64
# 下面每一项实验都只改动其中一个变量，其余与基线一致，这样才好比较。
# 说明：当前 torch 是 CPU 版（torch.cuda.is_available() = False），
#      无法统计显存，这里用「参数量」和「训练耗时」作为开销指标。
# ==================================================================
EXPERIMENT_EPOCHS = 60

def run_experiment(name, loaders, hidden_width=32, n_hidden=2, activation='relu',
                   lr=1e-2, opt_name='adam', epochs=EXPERIMENT_EPOCHS):
    set_seed(SEED)                      # 每个实验都从同样的随机状态出发
    tr, va, te = loaders
    net = MLP(hidden_width, n_hidden, activation).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    t0 = time.time()
    hist, ep, bacc, state = train_and_validate(
        net, tr, va, epochs=epochs, lr=lr, opt_name=opt_name,
        save_path=None, verbose=False)          # 不写文件，别覆盖主模型
    elapsed = time.time() - t0
    net.load_state_dict(state)
    _, te_acc = evaluate(net, te, nn.CrossEntropyLoss())
    return dict(name=name, history=hist, best_val_acc=bacc, test_acc=te_acc,
                n_params=n_params, time=elapsed)

base_loaders = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 64)

results = [
    run_experiment("基线 baseline",     base_loaders),
    run_experiment("隐藏层宽度 8",      base_loaders, hidden_width=8),
    run_experiment("隐藏层宽度 128",    base_loaders, hidden_width=128),
    run_experiment("隐藏层数 1",        base_loaders, n_hidden=1),
    run_experiment("激活函数 Tanh",     base_loaders, activation='tanh'),
    run_experiment("学习率 1e-3",       base_loaders, lr=1e-3),
    run_experiment("优化器 SGD",        base_loaders, opt_name='sgd'),
    run_experiment("batch size 16",     make_loaders(X_train, y_train, X_val, y_val,
                                                     X_test, y_test, 16)),
]

# 数据噪声也是可以对照的因素，但它会改变数据集本身，所以单独构造数据再跑。
for noise in (0.1, 0.4):
    Xn, yn = make_moons(n_samples=1000, noise=noise, random_state=SEED)
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(Xn, yn, test_size=0.2,
                                                  random_state=SEED, stratify=yn)
    Xn_tr, Xn_va, yn_tr, yn_va = train_test_split(Xn_tr, yn_tr, test_size=0.2,
                                                  random_state=SEED, stratify=yn_tr)
    sc = StandardScaler()
    Xn_tr, Xn_va, Xn_te = sc.fit_transform(Xn_tr), sc.transform(Xn_va), sc.transform(Xn_te)
    results.append(run_experiment(f"数据噪声 {noise}",
                                  make_loaders(Xn_tr, yn_tr, Xn_va, yn_va, Xn_te, yn_te, 64)))

print("\n【任务4】对照实验结果汇总（每个实验训练 %d 个 epoch）" % EXPERIMENT_EPOCHS)
print(f"{'实验':<22}{'参数量':>8}{'最佳Val Acc':>13}{'测试Acc':>10}{'耗时(s)':>10}")
print("-" * 66)
for r in results:
    print(f"{r['name']:<22}{r['n_params']:>8d}{r['best_val_acc']:>13.4f}"
          f"{r['test_acc']:>10.4f}{r['time']:>10.2f}")

# 把所有实验的验证曲线画在一张图里对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for r in results:
    axes[0].plot(r['history']['val_loss'], label=r['name'], linewidth=1.2)
    axes[1].plot(r['history']['val_acc'],  label=r['name'], linewidth=1.2)
axes[0].set_title("对照实验：验证集 Loss"); axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
axes[1].set_title("对照实验：验证集 Accuracy"); axes[1].set_xlabel("Epoch")
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("experiment_curves.png", dpi=150); plt.close(fig)

# 测试集准确率柱状图 + 参数量柱状图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
names = [r['name'] for r in results]
axes[0].bar(range(len(results)), [r['test_acc'] for r in results], color='tab:blue')
axes[0].set_xticks(range(len(results))); axes[0].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
axes[0].set_ylim(0.8, 1.01); axes[0].set_title("各实验的测试集准确率"); axes[0].grid(alpha=0.3, axis='y')
axes[1].bar(range(len(results)), [r['n_params'] for r in results], color='tab:orange')
axes[1].set_xticks(range(len(results))); axes[1].set_xticklabels(names, rotation=35, ha='right', fontsize=8)
axes[1].set_title("各实验的参数量"); axes[1].grid(alpha=0.3, axis='y')
fig.tight_layout(); fig.savefig("experiment_summary.png", dpi=150); plt.close(fig)

best_exp = max(results, key=lambda r: r['test_acc'])
print(f"\n其中测试集表现最好的是：【{best_exp['name']}】"
      f"（测试 Acc {best_exp['test_acc']:.4f}）")
print("提示：数据噪声那两项用的是不同的数据集，横向比较时要注意这一点。")
print("【任务4】已保存 experiment_curves.png 与 experiment_summary.png")

# ==================================================================
# 任务 5：混淆矩阵 + 判错样本分析
# ==================================================================
y_pred, y_proba = predict(loaded_model, test_loader)
cm = confusion_matrix(y_test, y_pred)

print("\n【任务5】测试集混淆矩阵（行=真实，列=预测）：")
print(f"            预测 0   预测 1")
print(f"  真实 0     {cm[0,0]:>6d}   {cm[0,1]:>6d}")
print(f"  真实 1     {cm[1,0]:>6d}   {cm[1,1]:>6d}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'], digits=4))

# 画混淆矩阵
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['预测 0', '预测 1']); ax.set_yticklabels(['真实 0', '真实 1'])
ax.set_title("混淆矩阵（测试集）"); ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=150); plt.close(fig)

# ---- 挑几个判错的样本做分析 ----
wrong_idx = np.where(y_pred != y_test)[0]
print(f"\n测试集共 {len(y_test)} 个样本，判错 {len(wrong_idx)} 个"
      f"（错误率 {len(wrong_idx) / len(y_test) * 100:.1f}%）")

correct_idx = np.where(y_pred == y_test)[0]
conf_wrong = np.abs(y_proba[wrong_idx, 1] - 0.5) if len(wrong_idx) else np.array([np.nan])
conf_right = np.abs(y_proba[correct_idx, 1] - 0.5) if len(correct_idx) else np.array([np.nan])
print(f"判错样本的 |P(类别1)-0.5| 平均值 = {np.nanmean(conf_wrong):.3f}；"
      f"判对样本 = {np.nanmean(conf_right):.3f}")
print("（这个值越小说明模型的判断越犹豫，越靠近 0.5 就越像“拿不准”）")

print("\n被判错的前 5 个样本（特征为标准化后的值）：")
for k, i in enumerate(wrong_idx[:5], 1):
    print(f"  错误样本 {k}: 索引 #{i:3d} | 特征 [{X_test[i, 0]:+.3f}, {X_test[i, 1]:+.3f}]"
          f" | 真实={y_test[i]} 预测={y_pred[i]}"
          f" | P(0)={y_proba[i, 0]:.3f} P(1)={y_proba[i, 1]:.3f}")

# 把判错的点单独标在决策边界图上
plot_decision_boundary(loaded_model, X_test, y_test, "decision_boundary_with_errors.png",
                       title="任务5：测试集判错样本的位置")
print("\n【任务5】已保存 confusion_matrix.png 与 decision_boundary_with_errors.png")
print("\n全部完成。生成的图片：moons_dataset.png, split_data.png, training_curves.png,")
print("decision_boundary.png, experiment_curves.png, experiment_summary.png,")
print("confusion_matrix.png, decision_boundary_with_errors.png")
