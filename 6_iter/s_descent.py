#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import sys

# Matrix and vector
A = np.array([[3, 0.8], [0.8, 1.2]])
# A=np.array([[2,0],[0,2]])       # Alternative best case: circle
b = np.array([4, 6])

# Initial guess and tolerance level
x = np.array([0, 0])
eps = 1e-10

# Storage for solutions
max_iters = 100
x_k = np.empty((100, 2))
x_k[0, :] = x

# Steepest descent algorithm
r = b-np.dot(A, x)
k = 1
while True:

    # Steepest descent algorithm steps
    w = np.dot(A, r)
    a = np.dot(r, r)/np.dot(r.T, w)
    x = x+a*r
    r = r-a*w

    # Store current solution
    x_k[k, :] = x

    # Check for too many iterations
    if k > 100:
        print("Too many iterations")
        sys.exit()

    # Check for convergence
    k += 1
    if np.linalg.norm(r) < eps:
        print("Converged in", k, "iterations")
        break

# Plot results - create contours of phi function
n = 100
xx = np.linspace(-4, 6, n)
yy = np.linspace(-2, 8, n)
X, Y = np.meshgrid(xx, yy)
pxy = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        u = np.array([X[i, j], Y[i, j]])
        pxy[i, j] = 0.5*np.dot(u.T, np.dot(A, u))-np.dot(u, b)

fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
# Plot contours
contourf = ax.contourf(X, Y, pxy, 10, alpha=.75)
contour = ax.contour(X, Y, pxy, 10, colors='black')
# Plot results: overlay progress of algorithm
ax.plot(x_k[:k, 0], x_k[:k, 1], color='red', marker='x', ms=3, lw=0.5, label='Steepest descent')

# Formatting
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend(loc='upper right', fontsize=8, labelcolor='red')
plt.show()