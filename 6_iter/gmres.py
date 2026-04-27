#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def arnoldi(A, b, n):
    # Initialize quantity
    m = A.shape[0]
    Q = np.zeros((m, n + 1))        # Q_n and Q_{n+1}
    Ht_n = np.zeros((n + 1, n))     # \tilde{H}_n
    Q[:, 0] = b / np.linalg.norm(b) # q_0 = b / ||b||
    
    # Arnoldi iteration loop
    for j in range(n):
        v = A @ Q[:, j]                     # v = A q_j
        for i in range(j + 1):
            Ht_n[i, j] = np.dot(Q[:, i], v) # h_ij = q_i^* v
            v = v - Ht_n[i, j] * Q[:, i]    # v = v - h_ij q_i
        Ht_n[j + 1, j] = np.linalg.norm(v)  # h_{j+1,j} = ||v||
        Q[:, j + 1] = v / Ht_n[j + 1, j]    # q_{j+1} = v / h_{j+1,j}

    return Q, Ht_n

def gmres(A, b, n_max):
    # Store the solution and residuals
    gmres_residuals = []
    x_n = np.zeros(A.shape[0])
    beta = np.linalg.norm(b)

    # gmres iteration loop
    for n in range(1, n_max + 1):
        # 1. Step n of Arnoldi iteration
        Q, Ht_n = arnoldi(A, b, n)

        # 2. Find y for min || \tilde{H}_n y - ||b|| e1 ||
        e1 = np.zeros(n + 1) # e_1 = [1, 0, ..., 0]^T
        e1[0] = beta         # ||b|| e_1
        y, *_ = np.linalg.lstsq(Ht_n, e1, rcond=None)

        # 3. Least squares solution x_n = Q_n y
        x_n = Q[:, :n] @ y
        residual = np.linalg.norm(b - A @ x_n) / beta
        gmres_residuals.append(residual)

        # 4. Check for convergence
        if residual < 1e-16:
            break

    return x_n, gmres_residuals

# Construct matrix A
m = 200
np.random.seed(42)
A = 2*np.eye(m) + 0.5/np.sqrt(200)*np.random.randn(m, m)
# Set the maximum number of iterations for GMRES
n_max = 10
# Right-hand side
b = np.ones(m)

# GMRES
x_n, gmres_residuals = gmres(A, b, n_max)

# Visualize
fig = plt.figure(figsize=(4, 2), dpi=300)
gs = GridSpec(1, 2, width_ratios=[0.5, 1.], figure=fig)
ax_main = fig.add_subplot(gs[0, 1])
ax_inset = fig.add_axes([0.45, 0.2, 0.2, 0.28])

# Plot eigenvalues of A in inset
eigs = np.linalg.eigvals(A)
real_eigs = np.real(eigs)
img_eigs = np.imag(eigs)
ax_inset.scatter(real_eigs, img_eigs, s=0.1, color='blue', alpha=0.5)
ax_inset.set_xlabel('$Re$', fontsize=5, labelpad=1)
ax_inset.set_ylabel('$Im$', fontsize=5, labelpad=1)
ax_inset.set_title('Eigenvalues of A', fontsize=5, pad=2)
# Prettify
ax_inset.set_xlim(-1, 3)
ax_inset.set_ylim(-1.5, 1.5)
ax_inset.axhline(y=0, color='k', linewidth=0.5)
ax_inset.axvline(x=0, color='k', linewidth=0.5)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_aspect('equal')

# Plot convergence in main panel
ax_main.semilogy(range(1, len(gmres_residuals) + 1), gmres_residuals, marker='o', label='GMRES residual')
ax_main.set_xlabel('Iteration $n$', fontsize=9)
ax_main.set_ylabel(r'$\frac{\|r_n\|_2}{\|b\|_2}$', fontsize=9, rotation=0)
ax_main.set_xlim(0, 10)
ax_main.set_ylim(1e-6, 1)
ax_main.grid(True, linewidth=0.5)
ax_main.tick_params(axis='both', which='major', labelsize=8)
# Plot reference convergence line 4^(-n)
ref_line = [4**(-n) for n in range(1, len(gmres_residuals) + 1)]
ax_main.semilogy(range(1, len(gmres_residuals) + 1), ref_line, color='red', linestyle='--', label=r'$4^{-n}$')
ax_main.legend(fontsize=8, loc='upper right')

# Save the figure
# plt.savefig("results/gmres_convergence_well.pdf", bbox_inches='tight', dpi=300)
plt.show()