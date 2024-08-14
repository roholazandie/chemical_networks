import numpy as np
import matplotlib.pyplot as plt
from golden_meas_shift import generate_random_golden_mean_shift_sequence
from collections import Counter


def read_sequence_from_file(filename):
    with open(filename, 'r') as file:
        sequence = file.read().strip()
    return sequence




def box_counting_dimension(sequence, box_sizes):
    counts = []
    length = len(sequence)

    for size in box_sizes:
        num_boxes = (length + size - 1) // size  # Number of boxes of this size
        boxes = [sequence[i * size:(i + 1) * size] for i in range(num_boxes)]
        # non_empty_boxes = sum(1 for box in boxes if 1 in box)
        non_empty_boxes = 2^size - Counter(boxes).get(0, 0)
        counts.append(non_empty_boxes)

    return counts


def calculate_fractal_dimension(box_sizes, counts):
    log_box_sizes = np.log(box_sizes)
    log_counts = np.log(counts)

    # Fit a line to the log-log plot
    coeffs = np.polyfit(log_box_sizes, log_counts, 1)
    fractal_dimension = -coeffs[0]

    # Plotting the log-log plot
    plt.plot(log_box_sizes, log_counts, 'o', label='Data points')
    plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), label=f'Fit: slope = {fractal_dimension:.2f}')
    plt.xlabel('log(Box size)')
    plt.ylabel('log(Count)')
    plt.legend()
    plt.show()

    return fractal_dimension


# Read the sequence from the file
input_filename = 'random_golden_mean_shift_sequence.txt'
# sequence = read_sequence_from_file(input_filename)
sequence = generate_random_golden_mean_shift_sequence(10**7)
# Define box sizes to use for the calculation
box_sizes = [2 ** i for i in range(1, 16)]  # Box sizes ranging from 2 to 2^16

# Perform box counting
counts = box_counting_dimension(sequence, box_sizes)

# Calculate the fractal dimension
fractal_dimension = calculate_fractal_dimension(box_sizes, counts)

print(f"The estimated fractal dimension of the sequence is {fractal_dimension:.2f}.")
