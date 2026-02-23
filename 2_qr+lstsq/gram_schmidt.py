#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def cgs(A):
    """
    Compute the QR factorization of a matrix A using the classical Gram-Schmidt process.
    """
    m, n = A.shape
    v = np.zeros((m, n), dtype=np.float64)
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        vj = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            vj = vj - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(vj)
        Q[:, j] = vj / R[j, j]

    return Q, R

def mgs(A):
    """
    Compute the QR factorization of a matrix A using the modified Gram-Schmidt process.
    """
    m, n = A.shape
    v = np.zeros((m, n), dtype=np.float64)
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        v[:, i] = A[:, i].copy()
    for i in range(n):
        R[i, i] = np.linalg.norm(v[:, i])
        Q[:, i] = v[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], v[:, j])
            v[:, j] = v[:, j] - R[i, j] * Q[:, i]

    return Q, R


### Demo 1 ###
# Make a random matrix A
seed = 42
dim = 3
np.random.seed(seed) # set random seed for reproducibility
A = np.random.rand(dim, dim)

# QR with classical Gram-Schmidt
Q_cgs, R_cgs = cgs(A)
print(f'Q (CGS):\n{Q_cgs}\n')
print(f'R (CGS):\n{R_cgs}\n')
# QR with modified Gram-Schmidt
Q_mgs, R_mgs = mgs(A)
print(f'Q (MGS):\n{Q_mgs}\n')
print(f'R (MGS):\n{R_mgs}\n')
# QR with NumPy's built-in function
Q_np, R_np = np.linalg.qr(A)
print(f'Q (NumPy):\n{Q_np}\n')
print(f'R (NumPy):\n{R_np}\n')

# Check the orthogonality of Q
orthogonality_cgs = np.allclose(Q_cgs.T @ Q_cgs, np.eye(dim))
orthogonality_mgs = np.allclose(Q_mgs.T @ Q_mgs, np.eye(dim))
print(f'Orthogonality of Q (CGS): {orthogonality_cgs}')
print(f'Orthogonality of Q (MGS): {orthogonality_mgs}')


### Demo 2 ###
# Make a random orthogonal matrix U
seed = 21
np.random.seed(seed)  # set random seed for reproducibility
U, _ = np.linalg.qr(np.random.rand(80, 80))
# Make a random orthogonal matrix V
seed = 16
np.random.seed(seed)  # set random seed for reproducibility
V, _ = np.linalg.qr(np.random.rand(80, 80))
# Make singular values from 2^-1 to 2^-80
S = np.diag(2.0**(-1*np.arange(1, 81, 1)))
# Construct A with SVD
A = U @ S @ V.T

# QR with classical Gram-Schmidt
Q_cgs, R_cgs = cgs(A)
# QR with modified Gram-Schmidt
Q_mgs, R_mgs = mgs(A)

fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)

# Plot singular values
ax.semilogy(np.diag(R_cgs), 'o', ms=4, color='tab:blue', mfc='none', label='Classical GS')
ax.semilogy(np.diag(R_mgs), 'x', ms=4, color='tab:orange', label='Modified GS')
# Plot errors and scaling line
fr = [2**(-k) for k in range(1, 81)]
ax.semilogy(fr, '--', color='tab:gray')
ax.hlines(1e-8, 0, 80, colors='tab:gray', linestyles='--', alpha=0.5)
ax.hlines(1e-16, 0, 80, colors='tab:gray', linestyles='--', alpha=0.5)

# Formatting
ax.set_ylabel('$r_{jj}$', fontsize=9)
ax.legend(loc='lower left', fontsize=9)
ax.text(8, 1e-8, '$\sqrt{\epsilon_{mach}}$', fontsize=8, color='tab:gray', ha='center', va='bottom', alpha=0.5)
ax.text(8, 10.0**(-15.5), '$\epsilon_{mach}$', fontsize=8, color='tab:gray', ha='center', va='bottom', alpha=0.5)
ax.text(62, 1e-22, '$2^{-j}$', fontsize=8, color='tab:gray', ha='center', va='bottom')
# Prettify
ax.set_xlim(0, 80)
ax.set_ylim(1e-25, 1)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', bottom=False, left=False)
ax.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/gram_schmidt.pdf', bbox_inches='tight', transparent=True)
plt.show()