#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D matrix
m = 30
n = 30
# Create a meshgrid for x and y
x = np.linspace(0, m-1, m)
y = np.linspace(0, n-1, n)
X, Y = np.meshgrid(x, y)
# Make a smooth Gaussian matrix
A = np.exp(-0.05 * ((X - 10)**2 + (Y - 20)**2))

fig = plt.figure(figsize=(6, 2), dpi=300)
# 2D heatmap
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(A, cmap='inferno')
# 3D surface plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, A, cmap='inferno', edgecolor='none', alpha=0.9, rstride=1, cstride=1)

# Formatting
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('2D Heatmap', fontsize=9)
ax1.set_xlabel('$x$', fontsize=8)
ax1.set_ylabel('$y$', fontsize=8)
ax2.set_title('3D Surface Plot', fontsize=9)
ax2.tick_params(axis='x', pad=-5)
ax2.tick_params(axis='y', pad=-5)
ax2.tick_params(axis='z', pad=-2)
ax2.set_xlabel('$x$', fontsize=8, labelpad=-7)
ax2.set_ylabel('$y$', fontsize=8, labelpad=-7)
ax2.set_zlabel('$z$', fontsize=8, labelpad=-5)
# Prettify
for a in [ax1, ax2]:
    a.tick_params(axis='both', which='major', labelsize=8)
    a.tick_params(axis='both', which='minor', bottom=False, left=False)
    a.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/vizmat_3d.pdf', bbox_inches='tight', transparent=True)
plt.show()