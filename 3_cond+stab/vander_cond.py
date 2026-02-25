#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

ns = np.arange(2, 100) # degree of the polynomial

# Loop through the values of n and create the Vandermonde matrix
# Calculate the condition number
kappas = []
for n in ns:
    x = np.linspace(0, 1, n) # n points in the interval [-1, 1]
    V = np.vander(x, increasing=True)
    kappa = np.linalg.cond(V)
    kappas.append(kappa)

# Plot the condition number as a function of n
fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
ax.plot(ns, kappas, label='Vandermonde matrix', color='tab:blue', lw=2)

# Formatting
# ax.set_ylim(1e2, 1e10)
ax.set_xlabel('Degree $n$', fontsize=9)
ax.set_ylabel('Condition number', fontsize=9)
ax.set_yscale('log')
ax.legend(loc='lower right', fontsize=9)
# Prettify
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', bottom=False, left=False)
ax.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/vander_cond.pdf', bbox_inches='tight', transparent=True)
plt.show()