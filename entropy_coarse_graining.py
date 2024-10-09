import numpy as np
import matplotlib.pyplot as plt


# Define a simple probability density function (e.g., a normal distribution)
def pdf(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


# Compute the entropy with discretization
def compute_entropy(h, x_min, x_max):
    # Discretize the interval [x_min, x_max] into bins of width h
    bins = np.arange(x_min, x_max, h)
    mid_points = bins + h / 2
    f_values = pdf(mid_points)

    # Normalize the pdf to ensure it sums to 1 in the discrete case
    f_values /= (f_values * h).sum()

    # First term: -\sum h * f(i h) * log(f(i h))
    entropy_discrete = -np.sum(h * f_values * np.log(f_values + 1e-10))  # +1e-10 to avoid log(0)

    # Second term: -\sum h * f(i h) * log(h)
    entropy_bin_size = -np.sum(h * f_values * np.log(h))

    # Total entropy
    total_entropy = entropy_discrete + entropy_bin_size
    return total_entropy


# Parameters
x_min = -5  # Lower bound of the range
x_max = 5  # Upper bound of the range
h_values = np.logspace(-5, -1, 50)  # Different bin sizes h from 0.001 to 0.1

# Compute the entropy for each h
entropies = [compute_entropy(h, x_min, x_max) for h in h_values]

# Plot the result
plt.figure(figsize=(8, 6))
plt.plot(h_values, entropies, marker='o')
plt.xscale('log')
plt.xlabel('Bin size h (log scale)')
plt.ylabel('Entropy H_h')
plt.title('Entropy H_h as a function of bin size h')
plt.grid(True)
plt.show()
