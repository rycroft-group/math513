#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Vandermonde interpolation function
n = 12
# Here the Vandermonde matrix is built in increasing order
def vander_func(x, b):
    fx = 0
    for j in range(n):
        fx += b[j] * x**j
    return fx

# Create data and a truncated Vandermonde matrix
x = np.linspace(0, 1, 50)
A = np.vander(x, n, increasing=True)
y = np.cos(4.*x)

# Solve using the built-in least squares function
b_lstsq = np.linalg.lstsq(A, y, rcond=None)[0]

# Solve using the normal equations
AT = np.transpose(A)
ATA = AT @ A
print(f'Condition number of A^T A: {np.linalg.cond(ATA)}')
b_noreq = np.linalg.solve(ATA, AT @ y)

# Evaluate the difference between the two parameter sets
print(f'Norm of difference between lstsq and normal equations: {np.linalg.norm(b_lstsq - b_noreq)}')

# Calculate the residuals
residual_lstsq = np.linalg.norm(A @ b_lstsq - y)
residual_noreq = np.linalg.norm(A @ b_noreq - y)
print(f'Residual norm (least squares): {residual_lstsq}')
print(f'Residual norm (normal equations): {residual_noreq}')

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
xnew = np.linspace(0, 1, 200)
vnew_lstsq = [vander_func(x, b_lstsq) for x in xnew]
vnew_noreq = [vander_func(x, b_noreq) for x in xnew]
ax.plot(x, y, 'o', ms=2, color='tab:blue', mfc='none', label='Data points')
ax.plot(xnew, vnew_lstsq, '-', lw=2, alpha=0.5, color='tab:orange', label='Least squares')
ax.plot(xnew, vnew_noreq, '-', color='tab:green', label='Normal equations')
# Formatting
ax.legend(loc='upper right', fontsize=9)
ax.set_xlabel('$x$', fontsize=9)
ax.set_ylabel('$f(x)$', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
# Prettify
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', bottom=False, left=False)
ax.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/lfit.pdf', bbox_inches='tight', transparent=True)
plt.show()