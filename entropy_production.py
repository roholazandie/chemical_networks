import random
import numpy as np
from collections import Counter


def generate_random_golden_mean_shift_sequence(length):
    random.seed(42)  # Set the random seed for reproducibility
    sequence = []
    prev = None  # Start with no previous symbol

    for _ in range(length):
        if prev == 1:
            next_symbol = 0  # If the previous symbol is 1, the next must be 0
        else:
            next_symbol = random.choice([0, 1])  # Otherwise, choose randomly between 0 and 1

        sequence.append(next_symbol)
        prev = next_symbol  # Update the previous symbol

    return sequence


def generate_all_possible_blocks(block_length):
    from itertools import product
    return list(product([0, 1], repeat=block_length))


def count_blocks(sequence, block_length):
    blocks = [tuple(sequence[i:i + block_length]) for i in range(len(sequence) - block_length + 1)]
    return Counter(blocks)


def calculate_entropy(block_counts, block_length):
    total_blocks = sum(block_counts.values())
    all_possible_blocks = generate_all_possible_blocks(block_length)

    probabilities = []
    for block in all_possible_blocks:
        if block in block_counts:
            count = block_counts[block]
            if total_blocks > 0:
                probability = count / total_blocks

            probabilities.append(probability)

    growth_rate = np.log2(len(block_counts)) / block_length
    print(f"Growth rate: {growth_rate}")

    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


def calculate_entropy_production(sequence, max_block_length):
    entropy_ratios = []
    block_lengths = range(1, max_block_length + 1)

    for n in block_lengths:
        block_counts = count_blocks(sequence, n)
        entropy = calculate_entropy(block_counts, n)
        entropy_ratio = entropy / n
        entropy_ratios.append(entropy_ratio)

    return block_lengths, entropy_ratios


# Generate the sequence
sequence_length = 1000000
sequence = generate_random_golden_mean_shift_sequence(sequence_length)

# Calculate entropy production for blocks up to a certain length
max_block_length = 20  # Adjust this value for longer block lengths if needed
block_lengths, entropy_ratios = calculate_entropy_production(sequence, max_block_length)

# Plot the entropy production ratio
import matplotlib.pyplot as plt

plt.plot(block_lengths, entropy_ratios, label='Entropy Production Ratio H(t)/t')
plt.xlabel('Block Length (n)')
plt.ylabel('Entropy Production Ratio H(t)/t')
plt.title('Entropy Production of Golden Mean Shift Sequence')
plt.legend()
plt.show()

# Print the final entropy production ratio
print(f"The final entropy production ratio at block length {max_block_length} is {entropy_ratios[-1]:.5f}")
