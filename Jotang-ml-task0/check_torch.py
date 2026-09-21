"""Check the Python/PyTorch environment and run a tensor operation."""

from importlib.metadata import PackageNotFoundError, version
import platform

import torch


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print("Environment information")
    print("-" * 50)
    print(f"Python version : {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build     : {torch.version.cuda or 'CPU-only wheel'}")
    print(f"CUDA available : {cuda_available}")
    print(f"Execution device: {device}")

    if cuda_available:
        print(f"GPU            : {torch.cuda.get_device_name(0)}")

    print("\nInstalled dependencies")
    print("-" * 50)
    for package in ("numpy", "matplotlib", "scikit-learn"):
        print(f"{package:14}: {package_version(package)}")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    result = a @ b

    print("\nTensor operation: matrix multiplication")
    print("-" * 50)
    print(f"a:\n{a.cpu()}")
    print(f"b:\n{b.cpu()}")
    print(f"a @ b:\n{result.cpu()}")

    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.equal(result.cpu(), expected), "Unexpected matrix multiplication result"
    print("\nTensor operation completed successfully.")


if __name__ == "__main__":
    main()
