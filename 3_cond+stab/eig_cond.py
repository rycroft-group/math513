#!/usr/bin/python3
import numpy as np

A = np.array([[1, 1000], 
              [0, 1]])
B = np.array([[1, 1000], 
              [0.001, 1]])

# Calculate eigenvalues
eigs_A = np.linalg.eigvals(A)
eigs_B = np.linalg.eigvals(B)
print(f'Eigenvalues of A: {eigs_A}')
print(f'Eigenvalues of B: {eigs_B}')

# Calculate condition numbers
kappa_A = np.linalg.cond(A)
kappa_B = np.linalg.cond(B)
print(f'Condition number of A: {kappa_A:.2e}')
print(f'Condition number of B: {kappa_B:.2e}')