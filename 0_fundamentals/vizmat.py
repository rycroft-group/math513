#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

# Create a random mxn matrix
m = 30
n = 20
seed = 42
np.random.seed(seed)  # set seed for reproducibility
A = np.random.randn(m, n)  # random matrix with normal distribution
# Set some entries to zero to make it sparse
A[A < 0.5] = 0

# Visualize matrix via different methods
fig, axes = plt.subplots(1, 4, figsize=(4.75, 2), dpi=300, 
                         gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
# spy will show the sparsity pattern
axes[0].spy(A)
# matshow will show the matrix as is
axes[1].matshow(A, cmap='viridis')
# imshow will also show the matrix as is
im = axes[2].imshow(A, cmap='viridis')

# Formatting
# Add a colorbar for the imshow plot
fig.colorbar(im, cax=axes[3])
axes[0].set_title('spy', fontsize=9)
axes[1].set_title('matshow', fontsize=9)
axes[2].set_title('imshow', fontsize=9)
# Prettify
for a in axes[:3]:
    a.set_xticks([])
    a.set_yticks([])
    a.tick_params(axis='both', which='major', labelsize=8)
    a.tick_params(axis='both', which='minor', bottom=False, left=False)
    a.tick_params(axis='both', which='both', right=False, top=False)
# Save figure
# plt.savefig('results/vizmat.pdf', bbox_inches='tight', transparent=True)
plt.show()