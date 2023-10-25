import random

def substitute_fasta_headers(fasta_file):
    sequences = []
    with open(fasta_file, 'r') as file:
        header = ""
        sequence = ""
        for line in file:
            if line.startswith('>'):
                if sequence != "":
                    sequences.append(sequence)
                    sequence = ""
                header = line.strip()
            else:
                sequence += line.strip()
    if sequence != "":
        sequences.append(sequence)

    # Substitute FASTA headers with the string "|endoftext|>"
    sequences = ["<|endoftext|>" + seq for seq in sequences]

    return sequences

def split_dataset(sequences, train_ratio):
    random.shuffle(sequences)
    num_sequences = len(sequences)
    num_train = int(num_sequences * train_ratio)
    train_set = sequences[:num_train]
    validation_set = sequences[num_train:]

    return train_set, validation_set

# Example usage
fasta_file = "./GFP_NCBI.fasta"  # Replace with the path to your FASTA file
train_ratio = 0.985  # Modify the ratio as needed (e.g., 0.9 for 90/10 split)
test_ratio = 0.0  # Modify the ratio as needed (e.g., 0.9 for 90/10 split)

# Step 1: Substitute FASTA headers
sequences = substitute_fasta_headers(fasta_file)

# Step 2: Split dataset
train_set, validation_set = split_dataset(sequences, train_ratio)
validation_set, test_set = split_dataset(validation_set, test_ratio)

# Save sequences to separate files
train_file = "train.txt"
validation_file = "validation.txt"
test_file = "test.txt"

with open(train_file, 'w') as train_output:
    train_output.write('\n'.join(train_set))

with open(validation_file, 'w') as validation_output:
    validation_output.write('\n'.join(validation_set))

with open(test_file, 'w') as test_output:
    test_output.write('\n'.join(test_set))

print("Train file saved as:", train_file)
print("Validation file saved as:", validation_file)
print("Test file saved as:", test_file)
