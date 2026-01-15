#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Make an arbitrary vector
x = np.array([1.2, 0.5, -0.1, 2.3, -1.05, -2.35])

# Calculate the p-norm for a range of values of p
pr = np.arange(1, 100)
nx = [np.linalg.norm(x, p) for p in pr]

# Calculate the infinity norm
ni = [np.linalg.norm(x, np.inf)] * len(pr)
# The multiplication by len(pr) is to make the
# lengths match for plotting

# Plot the norms
fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
ax.plot(pr, nx, label='$p$-norm', color='tab:blue', lw=2)
ax.plot(pr, ni, '--', label='$\infty$-norm', color='tab:orange', lw=2)

# Formatting
ax.set_ylim(2, 8)
ax.set_xlabel('$1 \leq p \leq 100$', fontsize=9)
ax.set_ylabel('$\|x\|_p$', fontsize=9)
ax.legend(loc='upper right', fontsize=9)
# Prettify
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', bottom=False, left=False)
ax.tick_params(axis='both', which='both', right=False, top=False)
plt.show()