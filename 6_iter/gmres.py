#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Construct matrix A
m = 200
np.random.seed(42)
A = 2*np.eye(m) + 0.5/np.sqrt(200)*np.random.randn(m, m)
# Set the maximum number of iterations for GMRES
n_max = 10
# Right-hand side
b = np.ones(m)

# Initialize quantity for GMRES
Q = np.zeros((m, n_max + 1))        # Q_n and Q_{n+1}
Ht_n = np.zeros((n_max + 1, n_max)) # \tilde{H}_n
Q[:, 0] = b / np.linalg.norm(b)     # q_0 = b / ||b||
# Store the solution and residuals
x_n = np.zeros(m)
gmres_residuals = []

# GMRES iteration loop
for n in range(n_max):
    # 1. Step n of Arnoldi iteration
    v = A @ Q[:, n]                     # v = A q_n
    for j in range(n+1):
        Ht_n[j, n] = np.dot(Q[:, j], v) # h_jn = q_j^* v
        v = v - Ht_n[j, n] * Q[:, j]    # v = v - h_jn q_j
    Ht_n[n + 1, n] = np.linalg.norm(v)  # h_{n+1,n} = ||v||
    Q[:, n + 1] = v / Ht_n[n + 1, n]    # q_{n+1} = v / h_{n+1,n}

    # 2. Find y for min || \tilde{H}_n y - ||b|| e1 ||
    rhs = np.zeros(n + 2)      # e_1 = [1, 0, ..., 0]^T
    rhs[0] = np.linalg.norm(b) # ||b|| e_1
    y, *_ = np.linalg.lstsq(Ht_n[:n + 2, :n + 1], rhs, rcond=None)

    # 3. GMRES iterate
    x_n = Q[:, :n + 1] @ y
    residual = np.linalg.norm(b - A @ x_n) / np.linalg.norm(b)
    gmres_residuals.append(residual)

    # 4. Check for convergence
    if residual < 1e-16:
        break

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