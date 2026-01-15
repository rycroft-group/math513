#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Define the level set function
# for the unit circle
# e.g. 2 norm is x^2 + y^2 - 1 = 0
def phi(x, y, p=2):
    if p == np.inf:
        return np.maximum(np.abs(x), np.abs(y)) - 1
    else:
        return (np.abs(x)**p + np.abs(y)**p)**(1/p) - 1

# Create meshgrid
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

# Compute vector norms
norm1 = phi(X, Y, p=1)
norm2 = phi(X, Y, p=2)
norm4 = phi(X, Y, p=4)
normi = phi(X, Y, p=np.inf)

# Optional: make gradient colors
# Create a color gradient for plotting
# Check out more colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
n = 7 # number of p-norms
colors = plt.cm.rainbow_r(np.linspace(0.,1.,n))

# Plot
fig, ax = plt.subplots(1, 4, figsize=(6, 2), sharey=True, dpi=300)
ax[0].contour(X, Y, norm1, levels=[0], colors=[colors[0]], linewidths=2)
ax[0].set_title('$1$-norm', fontsize=9)
ax[1].contour(X, Y, norm2, levels=[0], colors=[colors[1]], linewidths=2)
ax[1].set_title('$2$-norm', fontsize=9)
ax[2].contour(X, Y, norm4, levels=[0], colors=[colors[2]], linewidths=2)
ax[2].set_title('$4$-norm', fontsize=9)
ax[3].contour(X, Y, normi, levels=[0], colors=[colors[-1]], linewidths=2)
ax[3].set_title(r'$\infty$-norm', fontsize=9)

# Formatting
for a in ax:
    a.set_aspect('equal')
    a.grid(False)
    a.set_xlim([-1.25, 1.25])
    a.set_ylim([-1.25, 1.25])
    a.set_xticks([-1, 0, 1])
    a.set_yticks([-1, 0, 1])
    # Prettify
    a.tick_params(axis='both', which='major', labelsize=8)
    a.tick_params(axis='both', which='minor', bottom=False, left=False)
    a.tick_params(axis='both', which='both', right=False, top=False)
plt.show()