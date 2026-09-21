# Jotang ML Task 0

## Environment

- Python: 3.11.16
- Conda environment: `jotang-ml`
- PyTorch: 2.6.0 (CPU wheel)

## Install dependencies with the Tsinghua mirror

```cmd
conda activate jotang-ml
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Run the environment check

```cmd
set PYTHONUTF8=1
python check_torch.py
```

The script prints Python, PyTorch, NumPy, Matplotlib, and scikit-learn versions,
checks CUDA availability, and runs a tensor matrix multiplication.

## GPU note

For Windows, the Tsinghua PyPI mirror currently provides the CPU PyTorch wheel.
A CUDA-enabled wheel requires the PyTorch CUDA package index and a compatible
NVIDIA driver. This CPU installation is sufficient for Task 0 when GPU support
is not required.

## hello_ml.py 代码解析

### 1. `def analyze_scores(scores):`

定义函数 `analyze_scores`，把字典 `scores` 传入函数。

### 2. `if not scores:` 与 `raise ValueError("成绩数据不能为空")`

判断字典是否为空，若为空则提示「成绩数据不能为空」。

### 3. `sum(scores.values()) / len(scores)`

`sum` 求字典中成绩数值的总和，`len` 求字典中人数，相除得到平均成绩。

### 4. `key=scores.get`

表示按对应的值去比较（`max` 用它找出成绩最高的人）。
