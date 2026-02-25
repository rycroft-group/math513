#!/usr/bin/python3
from math import *
import numpy as np
import matplotlib.pyplot as plt

# Vandermonde interpolation function
n = 12 # degree-11
def vander_func(x, b):
    fx = 0
    for j in range(n):
        fx += b[j] * x**j
    return fx

# Construct rectangular Vandermonde matrix
x = np.linspace(0.2, 1, 5)
A = np.vander(x, n, increasing=True)
y = np.cos(4*x)

# Solve using the normal equations
AT = np.transpose(A)
ATA = np.dot(AT, A)
print('\nCondition number: ', np.linalg.cond(ATA))
b2 = np.linalg.solve(ATA, np.dot(AT, y))
print('Normal eqs. : Norm(r): ', np.linalg.norm(y-np.dot(A, b2)))

# Solve using least squares routine
b1=np.linalg.lstsq(A,y,rcond=None)[0]
print("lstsq solve : Norm(r): ",np.linalg.norm(y-np.dot(A,b1)))
print('Norm of b: ', np.linalg.norm(b1))

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=300)
ax.plot(x, y, 'x', ms=5, color='k', label='Data points')
# Python least squares
xnew = np.linspace(0, 1, 200)
vnew = [vander_func(x, b1) for x in xnew]
ax.plot(xnew, vnew, '-', lw=2, color='tab:orange', label='Library function')
# Regularization mu = 0.05
vnew_reg = [vander_func(x, b2) for x in xnew]
ax.plot(xnew, vnew_reg, '-', lw=2, color='tab:green', label='Regularization ($\mu=0.05$)')

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
# plt.savefig('results/under_lfit_compare.pdf', bbox_inches='tight', transparent=True)
plt.show()