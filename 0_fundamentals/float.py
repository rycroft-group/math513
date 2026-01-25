#!/usr/bin/python3
from decimal import Decimal

# First, let us print 0.1 and 0.2 respectively
print(0.1)
print(0.2)
# Then, let's print 0.1 + 0.2, and compare it to 0.3
print(0.1 + 0.2)
print((0.1 + 0.2) == 0.3)

# Next, let's properly print floating point numbers using the Decimal library
print(Decimal(0.1))
print(Decimal(0.2))
print(Decimal(0.1 + 0.2))
print(Decimal(0.3))