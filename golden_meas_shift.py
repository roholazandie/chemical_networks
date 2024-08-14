import random


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


def write_sequence_to_file(sequence, filename):
    with open(filename, 'w') as file:
        for item in sequence:
            file.write(str(item))


# Generate the first million elements of the random Golden Mean Shift sequence
sequence_length = 1000000
random_golden_mean_shift_sequence = generate_random_golden_mean_shift_sequence(sequence_length)

# Write the sequence to a text file
output_filename = 'random_golden_mean_shift_sequence.txt'
write_sequence_to_file(random_golden_mean_shift_sequence, output_filename)

print(
    f"The first {sequence_length} elements of the random Golden Mean Shift sequence have been written to {output_filename}.")
