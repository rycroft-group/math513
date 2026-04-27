#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt

# Matrix and vector
A = np.array([[3, 0.8], [0.8, 1.2]])
# A=np.array([[2,0],[0,2]])       # Alternative best case: circle
b = np.array([4, 6])

# Initial guess and tolerance level
x = np.array([0, 0])
eps = 1e-10

# Storage for solutions
max_iters = 100
x_k = np.empty((100, 2))
x_k[0, :] = x

# Conjugate gradient for Ax = b
def cg(A, b, x0, tol=1e-10, max_iters=100):
    x = x0.copy().astype(float)
    r = b - A @ x
    p = r.copy()
    rr_old = r @ r
    x_list = [x.copy()]

    for k in range(1, max_iters + 1):
        Ap = A @ p
        alpha = rr_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        x_list.append(x.copy())

        rr_new = r @ r
        if np.sqrt(rr_new) < tol:
            return x, np.array(x_list), k

        beta = rr_new / rr_old
        p = r + beta * p
        rr_old = rr_new

    return x, np.array(x_list), max_iters


# Run CG on the system from above: A u = f
x0 = np.zeros_like(b, dtype=float)
x_cg, x_cg_list, k_cg = cg(A, b, x0, tol=eps, max_iters=max_iters)

print("CG solution:", x_cg)
print("Iterations:", k_cg)
print("Residual norm:", np.linalg.norm(b - A @ x_cg))