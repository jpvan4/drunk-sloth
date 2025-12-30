import ast
import math
from fpylll import IntegerMatrix, LLL

def load_matrix_int(path):
    with open(path, "r") as f:
        txt = f.read()
    outer = ast.literal_eval(txt)
    mat = outer[0]
    n = len(mat)
    m = len(mat[0])
    A = IntegerMatrix(n, m)
    for i in range(n):
        for j in range(m):
            A[i, j] = int(mat[i][j])
    return A

def to_float_rows(A):
    return [[float(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)]

def check_lll_conditions(A, delta=0.99, eta=0.51, verbose=False):
    """
    Check LLL conditions directly:
      - |mu_{k,j}| <= eta for all k > j
      - delta * ||b*_{k-1}||^2 <= ||b*_k||^2 + mu_{k,k-1}^2 * ||b*_{k-1}||^2
    Using floating-point GSO (good enough for small 20x20 sanity testing).
    """
    n = A.nrows
    m = A.ncols

    # Convert to float row vectors
    B = to_float_rows(A)

    # GSO
    bstar = [[0.0]*m for _ in range(n)]
    Bnorm = [0.0]*n
    mu = [[0.0]*n for _ in range(n)]

    for k in range(n):
        # start from b_k
        for j in range(m):
            bstar[k][j] = B[k][j]

        for j in range(k):
            denom = Bnorm[j]
            if abs(denom) < 1e-18:
                mu[k][j] = 0.0
                continue
            dot = sum(B[k][t] * bstar[j][t] for t in range(m))
            mu[k][j] = dot / denom
            # subtract projection
            for t in range(m):
                bstar[k][t] -= mu[k][j] * bstar[j][t]

        Bnorm[k] = sum(x*x for x in bstar[k])
        mu[k][k] = 1.0

    # 1) size reduction: |mu_{k,j}| <= eta
    for k in range(n):
        for j in range(k):
            if abs(mu[k][j]) > eta + 1e-9:  # small tolerance
                if verbose:
                    print(f"Size-reduction fail at k={k}, j={j}, mu={mu[k][j]}")
                return False

    # 2) Lovász condition
    for k in range(1, n):
        lhs = Bnorm[k]
        rhs = (delta - mu[k][k-1]*mu[k][k-1]) * Bnorm[k-1]
        if lhs + 1e-9 < rhs:  # allow tiny FP wiggle
            if verbose:
                print(f"Lovasz fail at k={k}: lhs={lhs}, rhs={rhs}")
            return False

    return True

if __name__ == "__main__":
    delta = 0.99
    eta   = 0.51

    A_cpu = load_matrix_int("basis_original_20x20.basis")
    A_gpu = load_matrix_int("basis_gpu_20x20_int.basis")

    # CPU-reduced basis
    A_cpu_lll = IntegerMatrix(A_cpu.nrows, A_cpu.ncols)
    for i in range(A_cpu.nrows):
        for j in range(A_cpu.ncols):
            A_cpu_lll[i, j] = A_cpu[i, j]
    LLL.reduction(A_cpu_lll, delta=delta, eta=eta)

    print("CPU LLL basis satisfies LLL conditions?",
          check_lll_conditions(A_cpu_lll, delta, eta, verbose=True))

    print("GPU basis satisfies LLL conditions?",
          check_lll_conditions(A_gpu, delta, eta, verbose=True))
