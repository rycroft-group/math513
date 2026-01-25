#!/usr/bin/python3

h = 1

# Loop until h becomes too small
while h > 1e-20:

    # Check if 1+h==1 in floating point arithmetic
    if 1+h == 1:
        print(h, "  1+h=1")
    else:
        print(h, "  1+h!=1")

    h *= 0.1