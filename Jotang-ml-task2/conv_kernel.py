# -*- coding: utf-8 -*-
"""
Jotang ML 项目 · Task 2（猫狗分类）· 实践 1：卷积核

任务 1：数字图片如何表示（像素是什么 / 什么是 RGB / 灰度图与彩色图的区别）
任务 2：用 Pillow 读图 → NumPy 数组 → PyTorch Tensor，打印 shape / dtype / 最小最大
        / 某一个像素的 RGB，说明每个维度的含义，以及 HWC 与 CHW 的区别
任务 3：二维卷积的基本过程 —— 用图片里一个具体位置把计算过程摊开手算
任务 4：只用 Python + NumPy 手写二维卷积（不使用 cv2.filter2D / scipy.signal.convolve2d
        / torch.nn.Conv2d 等现成实现），并用纯循环版本交叉验证
任务 5：四种卷积核处理同一张图并放在一起比较
        （3×3 均值模糊 / 5×5 高斯模糊 / 锐化 / Sobel-X、Sobel-Y 及合成边缘图）
任务 6：解释每种处理带来的变化，并给出定量证据（高频能量）
任务 7：valid / same padding、stride 与输出尺寸的关系，实测值与公式对齐

运行方式（三选一）：
    1) conda activate jotang-ml   然后   python conv_kernel.py
    2) conda run -n jotang-ml python conv_kernel.py
    3) python conv_kernel.py --image 你自己的图片.jpg --outdir .

输出图片（默认放在本脚本同目录）：
    01_image_basics.png   数字图片的基本表示
    02_conv_kernels.png   四种卷积核的结果对比
    03_padding_stride.png valid / same / stride 的尺寸对比
    04_out_of_range.png   卷积结果越界后三种保存策略的对比
"""

# ==================================================================
# 【第 0 步】运行环境自举（关键，不要删）
# ------------------------------------------------------------------
# 现象：不激活 conda 环境而直接运行 python.exe 时，进程在做矩阵乘（np.dot / @）
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
# 真正调用 BLAS（np.dot / @，matplotlib 存图内部也会用到）时 MKL 再去初始化
# libomp.dll，就会直接抛 "OMP: Error #15" 并中止进程，而且没有任何 Python 报错。
# 这是 Intel/LLVM 官方给出的开关，只影响“允许多份 OpenMP 共存”，不改变数值结果
# （本脚本的卷积结果另有“纯 Python 循环实现”做交叉验证，不依赖 MKL）。
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np
import torch
from PIL import Image
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


def mat_text(m, prec=0, indent="    "):
    """把矩阵打印得整齐一点（每行都带同样的缩进）。"""
    s = np.array2string(np.asarray(m), precision=prec, suppress_small=True, separator=" ")
    return "\n".join(indent + line for line in s.splitlines())


# ==================================================================
# 一、图像的基本表示
# ==================================================================
def load_image(path):
    """读图：Pillow 负责解码，转成 NumPy 数组后就是普通数字矩阵。"""
    pil = Image.open(path).convert("RGB")
    arr = np.asarray(pil)                               # (H, W, 3) uint8
    return pil, arr


def describe_pixel(arr, y, x):
    """打印某个像素的 RGB 数值，并说明它对应图片里的什么位置。"""
    r, g, b = (int(v) for v in arr[y, x])
    print(f"  第 {y} 行第 {x} 列这个像素：R={r}  G={g}  B={b}")
    print("  它是一组 0~255 的整数，合起来决定这一个点的颜色"
          "（越接近 0 越暗、越接近 255 越亮）。")
    return r, g, b


# ==================================================================
# 二、手写二维卷积（任务 4）
# ==================================================================
def output_size(h, w, k, padding=0, stride=1):
    """输出尺寸公式：floor((H + 2p - k) / s) + 1 。"""
    return (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1


def conv2d(image, kernel, padding=0, stride=1, pad_mode="constant"):
    """手写二维卷积（只用了 NumPy 的切片 + 广播，没有任何现成卷积实现）。

    image  : (H, W) 灰度图，或 (H, W, C) 彩色图
    kernel : (kh, kw)，本实践里都是方阵
    padding: 每条边补多少圈 0（0 就是 valid，让核刚好贴住边）
    stride : 窗口每步滑几格
    pad_mode: "constant" 补 0（PyTorch 默认），"edge" 复制边缘像素

    做法：外层只遍历核里的每一个系数 (i, j)，一次性把整张图上"该系数负责的
    那些像素位置"用切片取出来（形状 Hout×Wout×C），乘上系数后累加。
    这等价于把 K×K 次"整图级的乘法"叠加起来，代替"每个输出像素单独算一遍"。
    """
    x = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    single = (x.ndim == 2)
    if single:
        x = x[:, :, None]                               # 统一成 (H, W, 1) 方便处理
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode=pad_mode)
    H, W, C = x.shape
    kh, kw = k.shape
    H_out, W_out = output_size(H, W, kh, 0, stride)
    out = np.zeros((H_out, W_out, C), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if k[i, j] == 0:
                continue                                # 系数是 0 就不必取窗口
            window = x[i:i + H_out * stride:stride, j:j + W_out * stride:stride, :]
            out += window * k[i, j]
    return out[:, :, 0] if single else out


def conv2d_naive(image, kernel, padding=0, stride=1):
    """纯 Python 多重循环版卷积：慢，只用来交叉验证上面那个向量化实现。
    逻辑最直白 —— 对每个输出像素，把核盖上去，对应位置相乘再求和。"""
    x = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    single = (x.ndim == 2)
    if single:
        x = x[:, :, None]
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    H, W, C = x.shape
    kh, kw = k.shape
    H_out, W_out = output_size(H, W, kh, 0, stride)
    out = np.zeros((H_out, W_out, C), dtype=np.float64)
    for oy in range(H_out):
        for ox in range(W_out):
            for c in range(C):
                s = 0.0
                for i in range(kh):
                    for j in range(kw):
                        s += x[oy * stride + i, ox * stride + j, c] * k[i, j]
                out[oy, ox, c] = s
    return out[:, :, 0] if single else out


# ==================================================================
# 三、四种卷积核（任务 5）
# ==================================================================
K_MEAN3 = np.ones((3, 3)) / 9.0                         # 3×3 均值模糊


def gaussian_kernel(size=5, sigma=1.0):
    """按二维高斯函数生成 size×size 的核，再归一化（各元素加起来 = 1）。
    G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))，中心为 0 点。"""
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


K_SHARPEN = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float64)

K_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float64)

K_SOBEL_Y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float64)

K_LAPLACE = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float64)


def hf_energy(gray):
    """高频能量：拉普拉斯结果的绝对平均值。数值越大说明细节/边缘越多。
    这是用来定量说明"模糊 = 高频被削弱、锐化 = 高频被放大"的指标。"""
    return float(np.abs(conv2d(gray, K_LAPLACE)).mean())


# ==================================================================
# 四、卷积结果的保存策略（任务 7 末尾那个提醒：越界怎么办）
# ==================================================================
def to_uint8(v, mode="clip", lo=None, hi=None):
    """把 float 卷积结果变成能保存成图片的 0~255 uint8。

    mode="clip"   ：直接截断到 [0, 255]（保留原始数值大小，是"如实保存"）
    mode="minmax" ：按当前图的最小/最大值线性拉伸到 0~255（对比度最好）
    mode="abs"    ：取绝对值后按最大值拉伸（边缘图常用，负边缘也有意义）
    """
    v = np.asarray(v, dtype=np.float64)
    if mode == "clip":
        out = np.clip(v, 0, 255)
    elif mode == "minmax":
        lo = v.min() if lo is None else lo
        hi = v.max() if hi is None else hi
        out = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo) * 255.0
    elif mode == "abs":
        a = np.abs(v)
        m = a.max()
        out = np.zeros_like(a) if m == 0 else a / m * 255.0
    else:
        raise ValueError(f"未知的 mode: {mode}")
    return np.clip(out, 0, 255).round().astype(np.uint8)


def to_gray(rgb):
    """把 (H, W, 3) 的彩色结果按亮度权重压回灰度，方便统一算高频能量。"""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim == 2:
        return rgb
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def main():
    ap = argparse.ArgumentParser(description="实践 1：卷积核（手写二维卷积 + 四种核对比）")
    ap.add_argument("--image", default=_os.path.join(HERE, "sample_highway.png"),
                    help="要处理的图片，默认用课件里那张公路图")
    ap.add_argument("--outdir", default=HERE, help="输出图片的目录")
    args = ap.parse_args()
    outdir = _os.path.abspath(args.outdir)
    _os.makedirs(outdir, exist_ok=True)

    pil, arr = load_image(args.image)
    gray = np.asarray(pil.convert("L"))
    H, W, C = arr.shape
    print(f"读入图片 : {args.image}")
    print(f"原始尺寸 : 宽 {W} × 高 {H}")

    # ==============================================================
    # 任务 1 & 2：数字图片的表示、HWC 与 CHW
    # ==============================================================
    section("任务 1 & 2 · 一张图片在计算机里到底是什么")
    print(f"1) NumPy 数组  shape = {arr.shape}   dtype = {arr.dtype}")
    print(f"   最小值 = {arr.min()}   最大值 = {arr.max()}   "
          f"（uint8，也就是每个通道 8 位、取值 0~255）")
    print(f"   shape 的三个数依次是：高 H={H}、宽 W={W}、通道 C={C} —— 这就是 HWC 排列。")
    print(f"   3 个通道分别记录红(R)、绿(G)、蓝(B)的强弱；每个数字是 0~255 的亮度。")
    y, x = H // 2, W // 2
    r, g, b = describe_pixel(arr, y, x)

    gray_pil = np.asarray(pil.convert("L"))
    gray_manual = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).round()
    print(f"2) 灰度图 shape = {gray_pil.shape}：只有 H、W 两个维度，"
          f"每个像素只剩一个亮度值（0=黑，255=白）。")
    print(f"   灰度 = 0.299R + 0.587G + 0.114B（人眼对绿色最敏感，所以 G 的权重最大）。")
    print(f"   手算灰度与 Pillow convert('L') 的最大差 = {np.abs(gray_manual - gray_pil).max():.0f} "
          f"（只差四舍五入）")

    # 注意：PIL 给的数组是只读的，torch 不能直接包只读数组（会警告），先复制一份
    t_hwc = torch.from_numpy(np.array(arr, copy=True))
    t_chw = t_hwc.permute(2, 0, 1)
    print(f"3) torch.from_numpy(arr).shape = {tuple(t_hwc.shape)}  依旧是 HWC")
    print(f"   .permute(2, 0, 1) 之后      = {tuple(t_chw.shape)}  这就是 CHW")
    print(f"   permute 只是换了看数据的“视角”，内存没动：is_contiguous = {t_chw.is_contiguous()}，"
          f"真正喂给 nn.Conv2d 前要 .contiguous()")
    print(f"   卷积层还要一个 batch 维（unsqueeze(0)）→ {tuple(t_chw.unsqueeze(0).shape)} = NCHW，"
          f"即 (样本数, 通道, 高, 宽)。")
    print("   为什么 PyTorch 用 CHW：卷积核是按“通道”分组做乘加的，把通道放最前面，"
          "取某通道的整张图就是连续的切片；\n   而 Pillow、OpenCV、matplotlib 直接按 HWC 理解图片，"
          "所以这两套顺序互相转换非常频繁。")

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    axes[0].imshow(arr)
    axes[0].add_patch(plt.Circle((x, y), 14, color="#ff3b30", fill=False, lw=2))
    axes[0].set_title(f"原图 HWC {arr.shape}\n红圈处像素 R={r} G={g} B={b}", fontsize=10)
    axes[1].imshow(gray_pil, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"灰度图 {gray_pil.shape}\n(只有一个亮度通道)", fontsize=10)
    for i, cname in enumerate("RGB"):
        axes[2 + i].imshow(arr[:, :, i], cmap="gray", vmin=0, vmax=255)
        axes[2 + i].set_title(f"{cname} 通道 (H, W)\n单看某一个通道", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("任务 1 & 2：像素、RGB 通道、灰度图", fontsize=13)
    fig.tight_layout()
    fig.savefig(_os.path.join(outdir, "01_image_basics.png"), dpi=150)
    plt.close(fig)

    # ==============================================================
    # 任务 3 & 4：卷积过程 + 手写实现
    # ==============================================================
    section("任务 3 & 4 · 二维卷积到底怎么算（手写实现 + 手算验证）")
    g = gray.astype(np.float64)
    sobel_x_full = conv2d(g, K_SOBEL_X)
    y0, x0 = (int(v) for v in np.unravel_index(int(np.abs(sobel_x_full).argmax()),
                                               sobel_x_full.shape))
    print(f"先在整张图上跑一遍 Sobel-X，挑出反应最强的位置：第 {y0} 行、第 {x0} 列 —— "
          f"那里正是图中竖直方向明暗跳变最剧烈的地方。")
    print("现在把这个位置的数手算一遍，看能不能对上程序的结果：")
    win = g[y0:y0 + 3, x0:x0 + 3]
    print(f"  取出的 3×3 窗口（第 {y0}~{y0 + 2} 行、第 {x0}~{x0 + 2} 列）=")
    print(mat_text(win))
    print("  竖直边缘核 Sobel-X =")
    print(mat_text(K_SOBEL_X))
    print("  窗口里每个像素 × 核里对应位置的系数 =")
    print(mat_text(win * K_SOBEL_X))
    manual = float((win * K_SOBEL_X).sum())
    print(f"  9 个乘积相加得到 {manual:.0f}，这就是特征图里 (第 {y0} 行, 第 {x0} 列) 那个像素的值。")
    print("  窗口往右挪一列重算一次，扫完整张图就得到一张特征图。")

    assert abs(sobel_x_full[y0, x0] - manual) < 1e-9
    print(f"  手写卷积函数在同一位置输出 {sobel_x_full[y0, x0]:.0f} → 与手算一致")

    print("\n交叉验证：向量化实现 vs 纯多重循环实现（在 48×48 的小图上跑，纯循环在大图上太慢）")
    crop = g[100:148, 100:148]
    k_rand = np.array([[0.1, 0.2, -0.3],
                       [0.4, 0.0, 0.5],
                       [-0.2, 0.3, 0.1]])
    fast = conv2d(crop, k_rand, padding=1)
    slow = conv2d_naive(crop, k_rand, padding=1)
    print(f"  padding=1, stride=1 : 最大差异 = {np.abs(fast - slow).max():.2e}")
    fast2 = conv2d(crop, k_rand, padding=0, stride=2)
    slow2 = conv2d_naive(crop, k_rand, padding=0, stride=2)
    print(f"  padding=0, stride=2 : 最大差异 = {np.abs(fast2 - slow2).max():.2e}，"
          f"输出 shape = {fast2.shape}")
    assert np.allclose(fast, slow) and np.allclose(fast2, slow2)
    print("  两个实现完全一致，说明手写的卷积实现是正确的。")

    # ==============================================================
    # 任务 5 & 6：四种卷积核
    # ==============================================================
    section("任务 5 & 6 · 四种卷积核处理同一张图，并解释变化")
    K_GAUSS5 = gaussian_kernel(5, sigma=1.0)
    print("5×5 高斯核（sigma=1，按二维高斯函数生成后归一化）：")
    print(mat_text(K_GAUSS5, prec=4, indent="  "))
    print(f"  核里所有元素之和 = {K_GAUSS5.sum():.6f}（归一化保证总和为 1，"
          f"所以模糊后整体亮度不会变）")
    print(f"  均值核 K_MEAN3 每个元素 = 1/9 = {1 / 9:.6f}，同样总和为 1；"
          f"区别在于高斯核给中心更大权重，越靠边越轻。")

    blur_mean = conv2d(arr, K_MEAN3, padding=1)
    blur_gauss = conv2d(arr, K_GAUSS5, padding=2)
    sharpen = conv2d(arr, K_SHARPEN, padding=1)
    gx = conv2d(g, K_SOBEL_X, padding=1)
    gy = conv2d(g, K_SOBEL_Y, padding=1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    print(f"\n各结果的尺寸：原图 {arr.shape[:2]}，均值模糊 {blur_mean.shape[:2]}，"
          f"高斯模糊 {blur_gauss.shape[:2]}，锐化 {sharpen.shape[:2]}，边缘图 {mag.shape[:2]}")
    print("  （用了 same padding，所以尺寸和原图一样大；valid 会小一圈，见任务 7）")

    names = ["原图", "均值模糊 3×3", "高斯模糊 5×5", "锐化", "Sobel 边缘图"]
    energies = [hf_energy(g), hf_energy(to_gray(blur_mean)),
                hf_energy(to_gray(blur_gauss)), hf_energy(to_gray(sharpen)), hf_energy(mag)]
    print("\n定量证据 —— 高频能量（拉普拉斯结果的绝对平均值，越大说明细节/边缘越多）：")
    for n, e in zip(names, energies):
        print(f"    {n:<14} 高频能量 = {e:7.3f}")
    print(f"  均值模糊把高频能量压到原来的 {100 * energies[1] / energies[0]:.1f}%，"
          f"高斯模糊压到 {100 * energies[2] / energies[0]:.1f}%（柔和地压，"
          f"不像均值那样留下砖块感），")
    print(f"  锐化则把它抬到 {100 * energies[3] / energies[0]:.1f}%，"
          f"边缘图本身高达 {energies[4] / energies[0]:.1f} 倍 —— 说明它只留下边缘、"
          f"把平坦区域全部归零。")

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    panels = [
        ("原图", to_uint8(arr), None),
        ("3×3 均值模糊（same）", to_uint8(blur_mean, "clip"), None),
        ("5×5 高斯模糊（same, sigma=1）", to_uint8(blur_gauss, "clip"), None),
        ("锐化 [[0,-1,0],[-1,5,-1],[0,-1,0]]", to_uint8(sharpen, "clip"), None),
        ("Sobel-X：竖直边缘", to_uint8(gx, "abs"), "gray"),
        ("Sobel-Y：水平边缘", to_uint8(gy, "abs"), "gray"),
        ("合成边缘 sqrt(Gx^2+Gy^2)", to_uint8(mag, "abs"), "gray"),
    ]
    for ax, (title, img, cm) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cm)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    ax_bar = axes.ravel()[7]
    ax_bar.bar(range(len(names)), energies, color="#4da3ff")
    ax_bar.set_xticks(range(len(names)))
    ax_bar.set_xticklabels(["原图", "均值\n模糊", "高斯\n模糊", "锐化", "边缘\n图"], fontsize=9)
    ax_bar.set_title("高频能量对比（越高越“锋利”）", fontsize=10)
    ax_bar.grid(alpha=0.25, axis="y")
    fig.suptitle("任务 5 & 6：四种卷积核处理同一张图片", fontsize=14)
    fig.tight_layout()
    fig.savefig(_os.path.join(outdir, "02_conv_kernels.png"), dpi=150)
    plt.close(fig)

    # ==============================================================
    # 任务 7：padding / stride 与输出尺寸
    # ==============================================================
    section("任务 7 · 卷积核大小、padding、stride 与输出尺寸的关系")
    print("公式：H_out = floor((H + 2p - k) / s) + 1（W 同理）")
    print(f"本图 H = {H}, W = {W}\n")
    header = f"  {'核 k':>4} {'padding p':>10} {'stride s':>9} {'公式输出 H×W':>16} {'实测输出 H×W':>16}   说明"
    print(header)
    print("  " + "-" * (len(header) - 2))
    combos = []
    for ksize in (3, 5):
        kern = np.ones((ksize, ksize)) / (ksize * ksize)
        for pad in (0, 1, 2):
            for st in (1, 2):
                if H + 2 * pad < ksize or W + 2 * pad < ksize:
                    continue
                if st > 1 and (H + 2 * pad - ksize) < st:
                    continue
                fh, fw = output_size(H, W, ksize, pad, st)
                real = conv2d(g, kern, padding=pad, stride=st)
                tag = "valid（不填充）" if pad == 0 else ("same（尺寸不变）"
                      if pad == (ksize - 1) // 2 and st == 1 else "")
                assert real.shape == (fh, fw), f"公式与实测不一致: {real.shape} vs {(fh, fw)}"
                print(f"  {ksize:>4} {pad:>10} {st:>9} {f'{fh}×{fw}':>16} "
                      f"{f'{real.shape[0]}×{real.shape[1]}':>16}   {tag}")
                combos.append((ksize, pad, st, fh, fw))
    print("\n  规律小结：")
    print("   1) k 越大，输出越小：不填充时输出边长 = H - k + 1，也就是总共少了 k-1 个像素"
          "（每边少 (k-1)/2，因为核必须完整落在图内）。")
    print("   2) padding=p 相当于在四周补 p 圈 0，输出边长 +2p，正好用来抵消核带来的缩小。")
    print("   3) 想保持尺寸不变（same），stride=1 时取 p = (k-1)/2 即可："
          "k=3 用 p=1，k=5 用 p=2。")
    print("   4) stride=s 让窗口一次跳 s 格，输出边长大约变成 1/s，"
          "它和池化一样是“降采样”的手段。")

    blur_same = conv2d(arr, K_MEAN3, padding=1)
    blur_valid = conv2d(arr, K_MEAN3, padding=0)
    blur_g5 = conv2d(arr, K_GAUSS5, padding=2)
    blur_stride2 = conv2d(arr, K_MEAN3, padding=1, stride=2)
    # 把 padding 放大到 10，两种填充方式的差别才会明显可见（same 的 p=1 太薄看不出来）
    blur_pad0 = conv2d(arr, K_MEAN3, padding=10, pad_mode="constant")
    blur_padE = conv2d(arr, K_MEAN3, padding=10, pad_mode="edge")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].imshow(to_uint8(blur_valid, "clip"))
    axes[0, 0].set_title(f"valid：k=3, p=0, s=1\n输出 {blur_valid.shape[1]}×{blur_valid.shape[0]}（比原图小一圈）", fontsize=10)
    axes[0, 1].imshow(to_uint8(blur_same, "clip"))
    axes[0, 1].set_title(f"same：k=3, p=1, s=1\n输出 {blur_same.shape[1]}×{blur_same.shape[0]}（和原图一样大）", fontsize=10)
    axes[0, 2].imshow(to_uint8(blur_g5, "clip"))
    axes[0, 2].set_title(f"same：k=5, p=2, s=1\n输出 {blur_g5.shape[1]}×{blur_g5.shape[0]}，模糊更重", fontsize=10)
    axes[1, 0].imshow(to_uint8(blur_stride2, "clip"))
    axes[1, 0].set_title(f"stride=2：k=3, p=1, s=2\n输出 {blur_stride2.shape[1]}×{blur_stride2.shape[0]}（约缩小一半）", fontsize=10)
    axes[1, 1].imshow(to_uint8(blur_pad0, "clip")[:120, :120])
    axes[1, 1].set_title("补 0（constant）：把 p 放大到 10 看左上角\n边上多出来的 0 把边缘拉暗", fontsize=10)
    axes[1, 2].imshow(to_uint8(blur_padE, "clip")[:120, :120])
    axes[1, 2].set_title("补边缘像素（edge）：同样 p=10\n边缘不会变暗（推荐做法）", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("任务 7：valid / same / stride 的输出尺寸与边缘变化", fontsize=14)
    fig.tight_layout()
    fig.savefig(_os.path.join(outdir, "03_padding_stride.png"), dpi=150)
    plt.close(fig)

    # ==============================================================
    # 越界数值怎么保存
    # ==============================================================
    section("补充 · 卷积结果越界（<0 或 >255）怎么保存")
    below = int((sharpen < 0).sum())
    above = int((sharpen > 255).sum())
    print(f"  锐化结果范围：最小 {sharpen.min():.1f}，最大 {sharpen.max():.1f}（原图只有 0~255）")
    print(f"  其中小于 0 的有 {below} 个像素，大于 255 的有 {above} 个，"
          f"合计占 {100 * (below + above) / sharpen.size:.1f}%")
    wrap_demo = np.array([-5.0, -1.0, 300.0]).astype(np.uint8)
    print(f"  如果直接 astype(np.uint8)，负数会回绕成接近 255 的亮斑，"
          f"例如 [-5.0, -1.0, 300.0] -> {wrap_demo.tolist()}（-5 变成了 251）")
    print("  三种正确做法：")
    print("   a) clip：np.clip(v, 0, 255) —— 如实保存原始卷积值，负值一律截到 0，"
          "最接近“网络里看到的数”")
    print("   b) minmax：按实际最小/最大值线性拉伸到 0~255 —— 对比度最好，"
          "但相对亮度关系会被改变")
    print("   c) abs：取绝对值再按最大值拉伸 —— 边缘图专用，因为负边缘同样是边缘")
    print("  实际训练 CNN 时不做这些缩放：让数值原样进入网络，"
          "由 BatchNorm / ReLU 等层去处理幅度，"
          "只有导出给人看的图片时才需要 clip 或归一化。")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    axes[0].imshow(to_uint8(arr))
    axes[0].set_title("原图", fontsize=10)
    axes[1].imshow(to_uint8(sharpen, "minmax"))
    axes[1].set_title("锐化结果（minmax 归一化）", fontsize=10)
    axes[2].imshow(to_uint8(sharpen, "clip"))
    axes[2].set_title(f"锐化结果（clip 到 0~255）\n低于 0 的 {below} 个像素被截断", fontsize=10)
    axes[3].imshow(to_uint8(mag, "abs"), cmap="gray")
    axes[3].set_title("边缘图（取绝对值后拉伸）", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("补充：卷积结果越界时的保存策略", fontsize=14)
    fig.tight_layout()
    fig.savefig(_os.path.join(outdir, "04_out_of_range.png"), dpi=150)
    plt.close(fig)

    section("完成")
    for f in ("01_image_basics.png", "02_conv_kernels.png",
              "03_padding_stride.png", "04_out_of_range.png"):
        p = _os.path.join(outdir, f)
        print(f"  已保存 {p}  ({_os.path.getsize(p) // 1024} KB)")


if __name__ == "__main__":
    main()
