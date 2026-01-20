#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def unit_circle(p, num_points=400):
    # Parameterize the unit circle in L^p norm
    theta = np.linspace(0, 2*np.pi, num_points)
    x = np.cos(theta)
    y = np.sin(theta)
    if p == np.inf:
        # infinity-norm
        n = num_points // 4  # points per edge
        # Top edge (y = 1)
        x_top = np.linspace(-1, 1, n)
        y_top = np.ones(n)
        # Right edge (x = 1)
        x_right = np.ones(n)
        y_right = np.linspace(1, -1, n)
        # Bottom edge (y = -1)
        x_bottom = np.linspace(1, -1, n)
        y_bottom = -np.ones(n)
        # Left edge (x = -1)
        x_left = -np.ones(n)
        y_left = np.linspace(-1, 1, n)
        # Concatenate edges
        x = np.concatenate([x_top, x_right, x_bottom, x_left])
        y = np.concatenate([y_top, y_right, y_bottom, y_left])
    else:
        # p-norm
        r = (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(-1/p)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
    return np.vstack([x, y])


# Linear transformation matrix
A = np.array([[1, 2],
              [0, 2]])
norms = [1, 2, 4, np.inf]

# Compute vector norms
norm1_o = unit_circle(1)
norm2_o = unit_circle(2)
norm4_o = unit_circle(4)
normi_o = unit_circle(np.inf)
norm1_A = A @ norm1_o
norm2_A = A @ norm2_o
norm4_A = A @ norm4_o
normi_A = A @ normi_o

# Optional: make gradient colors
# Create a color gradient for plotting
# Check out more colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
n = 7  # number of p-norms
colors = plt.cm.rainbow_r(np.linspace(0., 1., n))

# Plot
fig, ax = plt.subplots(1, 4, figsize=(6, 2), sharey=True, dpi=300)
ax[0].plot(norm1_o[0], norm1_o[1], color=colors[0], linewidth=2)
ax[0].plot(norm1_A[0], norm1_A[1], color=colors[0], linewidth=2, alpha=0.25)
ax[0].set_title('$1$-norm', fontsize=9)
ax[1].plot(norm2_o[0], norm2_o[1], color=colors[1], linewidth=2)
ax[1].plot(norm2_A[0], norm2_A[1], color=colors[1], linewidth=2, alpha=0.25)
ax[1].set_title('$2$-norm', fontsize=9)
ax[2].plot(norm4_o[0], norm4_o[1], color=colors[2], linewidth=2)
ax[2].plot(norm4_A[0], norm4_A[1], color=colors[2], linewidth=2, alpha=0.25)
ax[2].set_title('$4$-norm', fontsize=9)
ax[3].plot(normi_o[0], normi_o[1], color=colors[-1], linewidth=2)
ax[3].plot(normi_A[0], normi_A[1], color=colors[-1], linewidth=2, alpha=0.25)
ax[3].set_title(r'$\infty$-norm', fontsize=9)

# Formatting
for a in ax:
    a.set_aspect('equal')
    a.grid(False)
    a.set_xlim([-3.5, 3.5])
    a.set_ylim([-3.5, 3.5])
    a.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    a.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    # Prettify
    a.tick_params(axis='both', which='major', labelsize=8)
    a.tick_params(axis='both', which='minor', bottom=False, left=False)
    a.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/norm_unit_circle_image.pdf', bbox_inches='tight', transparent=True)
plt.show()