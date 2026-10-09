# -*- coding: utf-8 -*-
"""
为 note.md 里两个“动手试试”的小问题做实验：
   实验 A：类别数量很不均衡时，模型会训成什么样？
   实验 B：怎样人为制造“过拟合”，它和正常训练有何不同？

运行：python note_experiments.py
只生成图片和打印数值，不影响 MLP.py 的主流程。
"""

# 第 0 步：环境自举（同 MLP.py，防止 numpy/MKL 原生崩溃）
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden_width=32, n_hidden=2, out_dim=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_width), nn.ReLU()]
            d = hidden_width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size=64, shuffle=True):
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def fit(model, tr_loader, va_loader=None, epochs=100, lr=1e-2):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        tl = tc = tt = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
            tc += (out.argmax(1) == yb).sum().item()
            tt += xb.size(0)
        hist["train_loss"].append(tl / tt)
        hist["train_acc"].append(tc / tt)
        if va_loader is not None:
            model.eval()
            vl = vc = vt = 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = crit(out, yb)
                    vl += loss.item() * xb.size(0)
                    vc += (out.argmax(1) == yb).sum().item()
                    vt += xb.size(0)
            hist["val_loss"].append(vl / vt)
            hist["val_acc"].append(vc / vt)
    return hist


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ==================================================================
# 实验 A：类别极不均衡（100 : 1000）
# ==================================================================
print("=" * 70)
print("实验 A：类别不均衡的数据集")
print("=" * 70)
set_seed(SEED)
Xa, ya = make_moons(n_samples=2000, noise=0.3, random_state=SEED)

idx_minority = np.where(ya == 0)[0][:40]       # 类别 0 只保留 40 个
idx_majority = np.where(ya == 1)[0]            # 类别 1 保留全部 1000 个
idx = np.concatenate([idx_minority, idx_majority])
np.random.default_rng(SEED).shuffle(idx)
Xa, ya = Xa[idx], ya[idx]
n0, n1 = int((ya == 0).sum()), int((ya == 1).sum())
print(f"数据集构成：类别 0（少数）= {n0} 个，类别 1（多数）= {n1} 个，"
      f"比例约 1 : {n1 / n0:.0f}")

Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    Xa, ya, test_size=0.3, random_state=SEED, stratify=ya)
sc = StandardScaler()
Xa_tr = sc.fit_transform(Xa_tr)
Xa_te = sc.transform(Xa_te)

model_a = MLP(hidden_width=32, n_hidden=2).to(device)
fit(model_a, make_loader(Xa_tr, ya_tr, 64), epochs=100, lr=1e-2)
pred_a = predict(model_a, make_loader(Xa_te, ya_te, 64, shuffle=False))

cm_a = confusion_matrix(ya_te, pred_a)
print("\n测试集混淆矩阵（行 = 真实 0/1，列 = 预测 0/1）：")
print(cm_a)
print(f"\n整体准确率            = {(pred_a == ya_te).mean():.4f}")
print(f"“无脑全猜多数类”的准确率 = {(ya_te == 1).mean():.4f}   <-- 对比用")
for c in (0, 1):
    mask = ya_te == c
    name = "少数类 0" if c == 0 else "多数类 1"
    print(f"  {name} 的召回率 = {(pred_a[mask] == c).mean():.4f}（该类共 {mask.sum()} 个）")
print("\n分类报告：")
print(classification_report(ya_te, pred_a,
                            target_names=["Class 0（少数）", "Class 1（多数）"], digits=4))

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cm_a, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm_a[i, j]), ha="center", va="center", fontsize=16,
                color="white" if cm_a[i, j] > cm_a.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["预测 0", "预测 1"]); ax.set_yticklabels(["真实 0", "真实 1"])
ax.set_title(f"类别不均衡（{n0} : {n1}）时的混淆矩阵")
ax.set_xlabel("预测标签"); ax.set_ylabel("真实标签")
fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig("note_imbalanced_cm.png", dpi=150); plt.close(fig)
print("\n已保存 note_imbalanced_cm.png")


# ==================================================================
# 实验 B：故意制造过拟合
# ==================================================================
print("\n" + "=" * 70)
print("实验 B：用极少的样本 + 很大的网络 + 很长的训练，制造过拟合")
print("=" * 70)
set_seed(SEED)
Xb, yb = make_moons(n_samples=400, noise=0.25, random_state=SEED)

# 只用 30 个样本训练，剩下的留作验证（不标准化，避免再引入别的变量）
Xb_tr, Xb_rest, yb_tr, yb_rest = train_test_split(
    Xb, yb, train_size=30, random_state=SEED, stratify=yb)
Xb_va, _, yb_va, _ = train_test_split(
    Xb_rest, yb_rest, test_size=0.5, random_state=SEED, stratify=yb_rest)
print(f"训练集只有 {len(yb_tr)} 个样本，验证集 {len(yb_va)} 个样本")
print("网络：3 个隐藏层 × 256 宽，Adam lr=1e-2，训练 800 个 epoch（不加任何正则化）")

big = MLP(hidden_width=256, n_hidden=3).to(device)
n_params = sum(p.numel() for p in big.parameters())
print(f"参数量 = {n_params}")
hist = fit(big, make_loader(Xb_tr, yb_tr, 16),
           make_loader(Xb_va, yb_va, 16, shuffle=False), epochs=800, lr=1e-2)

best_ep = int(np.argmin(hist["val_loss"])) + 1
print(f"\n训练到第 {best_ep} 个 epoch 时验证损失最低（{min(hist['val_loss']):.4f}，"
      f"此时验证准确率 {hist['val_acc'][best_ep - 1]:.4f}）")
print(f"训练结束时：训练 Loss = {hist['train_loss'][-1]:.4f}，"
      f"训练 Acc = {hist['train_acc'][-1]:.4f}")
print(f"            验证 Loss = {hist['val_loss'][-1]:.4f}，"
      f"验证 Acc = {hist['val_acc'][-1]:.4f}")
print(f"训练集与验证集的准确率差距 = "
      f"{hist['train_acc'][-1] - hist['val_acc'][-1]:+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(hist["train_loss"], label="训练 Loss")
axes[0].plot(hist["val_loss"], label="验证 Loss")
axes[0].axvline(best_ep - 1, color="gray", linestyle="--", linewidth=1)
axes[0].set_title("过拟合：训练 loss 一路下降，验证 loss 先降后升")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(hist["train_acc"], label="训练 Acc")
axes[1].plot(hist["val_acc"], label="验证 Acc")
axes[1].axvline(best_ep - 1, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("过拟合：训练准确率到 100%，验证准确率却跟不上")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("note_overfit_curves.png", dpi=150); plt.close(fig)

# 同一份小数据、换成小网络 + 少 epoch 做对照
set_seed(SEED)
small = MLP(hidden_width=8, n_hidden=1).to(device)
hist_s = fit(small, make_loader(Xb_tr, yb_tr, 16),
             make_loader(Xb_va, yb_va, 16, shuffle=False), epochs=200, lr=1e-2)
print(f"\n对照（小模型 1 层 × 8 宽，200 epoch）："
      f"训练 Acc {hist_s['train_acc'][-1]:.4f} / 验证 Acc {hist_s['val_acc'][-1]:.4f}，"
      f"差距 {hist_s['train_acc'][-1] - hist_s['val_acc'][-1]:+.4f}")

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(hist["train_acc"], label="大模型 训练 Acc")
ax.plot(hist["val_acc"], label="大模型 验证 Acc")
ax.plot(hist_s["train_acc"], label="小模型 训练 Acc", linestyle="--")
ax.plot(hist_s["val_acc"], label="小模型 验证 Acc", linestyle="--")
ax.set_title("大模型严重过拟合 vs 小模型相对正常")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("note_overfit_compare.png", dpi=150); plt.close(fig)
print("\n已保存 note_overfit_curves.png 与 note_overfit_compare.png")
