import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import math
import seaborn as sns
from entropy_production import generate_random_golden_mean_shift_sequence

# Function to generate Golden Mean Shift sequences of a given length
def golden_mean_shift_sequences(n):
    sequences = ['']
    while any(len(seq) < n for seq in sequences):
        sequences = [seq + '0' for seq in sequences if len(seq) < n] + \
                    [seq + '1' for seq in sequences if len(seq) < n and seq[-1:] != '1']
    return [seq for seq in sequences if len(seq) == n]


def generate_all_possible_blocks(block_length):
    from itertools import product
    return list(product([0, 1], repeat=block_length))


def reshape_to_square_grid(data):
    # Calculate the largest side length that can form a perfect square
    side_length = int(math.sqrt(len(data)))
    largest_square_length = side_length ** 2

    # Trim the data to fit into a perfect square grid
    trimmed_data = data[:largest_square_length]

    # Reshape the trimmed 1D array into a 2D array (square grid)
    trimmed_data = np.array(trimmed_data)
    square_grid = trimmed_data.reshape((side_length, side_length))

    return square_grid

# Function to create a near-square grid for the sequences
def create_square_grid(sequences, max_exponent):
    index = 0
    data = [0] * 2** max_exponent

    blocks = generate_all_possible_blocks(max_exponent)
    blocks = [''.join(map(str, block)) for block in blocks]
    for i, block in enumerate(blocks):
        # count the number of blocks in the sequence and fill the data array
        for _ in range(sequences.count(block)):
            data[index] = 1
            index += 1
    # shuffle(data)
    print(f"the ration of golden sequences is {sum(data)/len(data)}")
    data = reshape_to_square_grid(data)
    return data


# Function to plot sequences using a heatmap
# Function to plot sequences using a heatmap
# def plot_sequences_2d(data, exponent):
#     plt.figure(figsize=(10, 10))
#     sns.heatmap(data, cmap='viridis', cbar=False, linewidths=0.5, linecolor='black', square=True, xticklabels=False,
#                 yticklabels=False)
#
#     plt.title(f'Golden Mean Shift Sequences for all elements up to 2^{exponent}={2**exponent} elements')
#     plt.savefig(f'golden_mean_shift_sequences_{exponent}.png', dpi=300, bbox_inches='tight')
#
#     plt.show()


def plot_sequences_2d(data, exponent):
    plt.figure(figsize=(10, 10))

    # Create a color mesh plot
    plt.pcolormesh(data, cmap='viridis', edgecolors='none')

    plt.title(f'Golden Mean Shift Sequences for all elements up to 2^{exponent}={2 ** exponent} elements')
    plt.axis('equal')  # Ensure squares in the plot
    plt.axis('off')  # Hide the axis

    plt.savefig(f'golden_mean_shift_sequences_{exponent}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sequences_1d(data, exponent):
    plt.figure(figsize=(10, 2))
    sns.heatmap(data.reshape(1, -1), cmap='Greys', cbar=False, linewidths=0.5, linecolor='black',
                xticklabels=False, yticklabels=False)

    plt.xlabel('Position in sequence')
    plt.title('Golden Mean Shift Sequences')
    plt.savefig(f'golden_mean_shift_sequences_{exponent}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Define the lengths of the sequences as powers of 2
max_exponent = 10

sequence = generate_random_golden_mean_shift_sequence(2**max_exponent)
all_sequences = [tuple(sequence[i:i + max_exponent]) for i in range(1, len(sequence) - max_exponent + 1, max_exponent)]
all_sequences = [''.join(map(str, block)) for block in all_sequences]

# Create the square grid
data = create_square_grid(all_sequences, max_exponent)
# Plot the sequences
plot_sequences_2d(data, max_exponent)
