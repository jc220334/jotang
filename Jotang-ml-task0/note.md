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
