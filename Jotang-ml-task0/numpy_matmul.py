"""Create two NumPy matrices and perform matrix multiplication."""

import numpy as np


a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

b = np.array([
    [7, 8],
    [9, 10],
    [11, 12],
])

result = a @ b
expected = np.array([
    [58, 64],
    [139, 154],
])

print(f"numpy version: {np.__version__}")
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print("a =")
print(a)
print("b =")
print(b)
print("a @ b =")
print(result)
print(f"result shape: {result.shape}")

assert a.shape[1] == b.shape[0], "Matrix dimensions do not match"
assert np.array_equal(result, expected), "Unexpected matrix multiplication result"
print("Matrix multiplication completed successfully.")
