import numpy as np
from numpy.linalg import matrix_power
from numpy.linalg import eig

a = np.array([[1, 1],[1, 0]])
b = np.array([[0, 1, 1],[1, 0, 0],[1, 1, 0]])

# a and b are equivariant under fibration symmetry

# the same largest eignevalue => same entropy
print(eig(a))
print(eig(b))

# both generate Fibonacci sequences
a_powered = matrix_power(a, 11)
b_powered = matrix_power(b, 11)

assert a_powered[0][1] == b_powered[0][1]
print(a_powered)
print(b_powered)