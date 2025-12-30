# compare_lll.py
import ast
from fpylll import IntegerMatrix, LLL
import math

def load_matrix_int(path):
    with open(path, "r") as f:
        txt = f.read()
    # txt looks like: [[ [a11, ..., a1m], [a21, ..., a2m], ... ]]
    outer = ast.literal_eval(txt)
    mat = outer[0]
    n = len(mat)
    m = len(mat[0])
    A = IntegerMatrix(n, m)
    for i in range(n):
        for j in range(m):
            A[i, j] = int(mat[i][j])
    return A

def is_lll_reduced(A, delta=0.99, eta=0.51):
    # fpylll has a helper, but we'll call the reduction and compare to itself.
    # If A is already reduced, reduction should be idempotent.
    B = IntegerMatrix(A.nrows, A.ncols)
    for i in range(A.nrows):
        for j in range(A.ncols):
            B[i, j] = A[i, j]

    LLL.reduction(B, delta=delta, eta=eta)
    # Check if A and B are identical
    for i in range(A.nrows):
        for j in range(A.ncols):
            if A[i, j] != B[i, j]:
                return False
    return True

def norm_sq(v):
    return sum(int(x) * int(x) for x in v)

def row_as_list(A, i):
    return [int(A[i, j]) for j in range(A.ncols)]

if __name__ == "__main__":
    delta = 0.99
    eta   = 0.51

    # 1. Load original basis
    A = load_matrix_int("basis_original_20x20.basis")
    print("Original basis:")
    print("  n =", A.nrows, "m =", A.ncols)

    # 2. Compute CPU LLL via fpylll
    A_cpu = IntegerMatrix(A.nrows, A.ncols)
    for i in range(A.nrows):
        for j in range(A.ncols):
            A_cpu[i, j] = A[i, j]

    print("Running fpylll LLL...")
    LLL.reduction(A_cpu, delta=delta, eta=eta)

    # 3. Load GPU integer basis (rounded)
    A_gpu = load_matrix_int("basis_gpu_20x20_int.basis")

    # 4. Check if GPU basis is LLL-reduced
    print("\nChecking GPU basis LLL conditions (via idempotence test)...")
    gpu_lll_ok = is_lll_reduced(A_gpu, delta=delta, eta=eta)
    print("  GPU basis LLL-reduced? ->", gpu_lll_ok)

    # 5. Compare length profiles
    def length_profile(A):
        return [norm_sq(row_as_list(A, i)) for i in range(A.nrows)]

    cpu_profile = length_profile(A_cpu)
    gpu_profile = length_profile(A_gpu)

    print("\nCPU LLL vector norms:")
    print(cpu_profile)
    print("\nGPU LLL (rounded) vector norms:")
    print(gpu_profile)

    # 6. Simple GH-ish comparison (just sum of logs of norms)
    cpu_log_sum = sum(math.log(float(v)) for v in cpu_profile)
    gpu_log_sum = sum(math.log(float(v)) for v in gpu_profile)

    print("\nSum log(norm^2):")
    print("  CPU:", cpu_log_sum)
    print("  GPU:", gpu_log_sum)

    # 7. Optional: compare first few rows explicitly
    print("\nFirst 3 CPU LLL basis vectors:")
    for i in range(min(3, A_cpu.nrows)):
        print("  ", row_as_list(A_cpu, i))

    print("\nFirst 3 GPU LLL (rounded) basis vectors:")
    for i in range(min(3, A_gpu.nrows)):
        print("  ", row_as_list(A_gpu, i))
