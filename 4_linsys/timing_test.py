#!/usr/bin/python
from time import time, process_time

# Code credit: https://github.com/chr1shr/am205_examples/blob/master/2_num_lin_alg/timing_test.py

# Get the current time(), a measure of "wall-clock time", the physical passage of
# time as you would observe by looking at a wall clock
t0 = time()

# Get the current process_time(), a measure of the time that this program has actually
# spent on the processor
p0 = process_time()

# Carry out a large, arbitrary calculation
k = 0
for i in range(8000):
    for j in range(8000):
        k += i+j

# Print the elapsed time, using both the time() routine and the process_time() routine
ttotal = time()-t0
ptotal = process_time()-p0
print("time: {0:6.3f} s, process_time: {1:6.3f} s, ratio: {2:6.3f}".format(
    ttotal, ptotal, ttotal/ptotal))
