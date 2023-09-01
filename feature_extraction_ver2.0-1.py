from itertools import product
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py

# Read csv data
data = pd.read_csv('biogrid_training_database.csv')
output_file = 'biogrid_extracted_features2.hdf5'

# Features extraction functions
def calculate_amino_acid_proportion(sequence):
    sequence=str(sequence)
    total_length = len(sequence)
    proportion = [sequence.count(str(i)) / total_length for i in range(1, 6)]
    length=len(sequence)/1000
    return proportion,length

def get_all_n_mers(n):
    amino_acids = ['1', '2', '3', '4', '5']
    all_n_mers = [''.join(p) for p in product(amino_acids, repeat=n)]  # Use product to generate all possible N-Mers
    return all_n_mers

def calculate_frequencies(sequence, all_n_mers):
    sequence=str(sequence)
    # Count the number of occurrences of all n-mers in the sequence
    n_mer_counts = [sequence.count(mer) for mer in all_n_mers]
    # Convert to frequency
    total_n_mer_counts = sum(n_mer_counts)
    n_mer_frequencies = [count / total_n_mer_counts for count in n_mer_counts]
    return n_mer_frequencies

def calculate_nmer_position_percentage(sequence, n):
    sequence=str(sequence)
    n_mer_list = get_all_n_mers(n)
    counts_list = []
    
    def count_n_mers(sequence,mer,n):
        counts=[0]*20
        start = 0
        while start < len(sequence):
            start_index = sequence.find(mer, start) # Locate and record the position of the first letter of n-mer
            if start_index == -1:  # If the n-mer is not found, end the loop
                break
            end_index = start_index + len(mer) - 1  # Calculate the position of the last letter of n-mer
            start_position = start_index / len(sequence)
            end_position = end_index / len(sequence)
            if start_position >= 1:
                start_position = 0.99999
            if end_position >= 1:
                end_position = 0.99999
            start_percentage = int(20 * start_position)  # Calculate the percentage of these mer's first and last letters relative to the length of the sequence
            end_percentage = int(20 * end_position)
            if start_percentage == end_percentage:
                counts[start_percentage] += 1
            for i in range(start_percentage, end_percentage):
                counts[i] += 1
            start = start_index + 1  # Update the starting position to find the next subsequence in the next iteration

            # Do normalization if n=1
        if n == 1:
            total = sum(counts)
            if total != 0:
                counts = [10*(i/total) for i in counts]
        return counts
    
    # Create a list to store the counts for each n-mer
    for n_mer in n_mer_list:
        con = count_n_mers(sequence, n_mer, n)
        counts_list.append(con)
    
    return counts_list

def APAAC(sequence, maxlag):
    sequence = str(sequence)
    def calculate_h_values(sequence, H):
        sequence = list(map(int, str(sequence)))
        h_values = []
        hdiff_matrix = []
        for d in range(1, maxlag+1):
            hdiff_in_lengths = [0]*20
            sum_of_h = 0
            blur = 0
            item = 0
            # Calculate sum of H difference
            for i in range(len(sequence) - d):
                # If 5 is present in the sequence, blur the 5 related d value in sum
                if sequence[i] == 5 or sequence[i+d] == 5:
                    blur += 1
                    h_val = sum_of_h/item
                else:
                    h_val = H[sequence[i]-1] * H[sequence[i+d]-1]
                    item += 1
                    sum_of_h += h_val

            sum_of_h = blur*(sum_of_h/item) + sum_of_h
            ave_h = sum_of_h/(blur + item)
            
            # Calculate H difference in a length percentage list
            for k in range(len(sequence) - d):
                length_percentage = k / (len(sequence) - d)
                if length_percentage >= 1:
                    length_percentage = 0.99999
                if sequence[i] == 5 or sequence[i+d] == 5:
                    h_val = ave_h
                else:
                    h_val = H[sequence[i]-1] * H[sequence[i+d]-1]
                hdiff_in_lengths[int(20 * length_percentage)] += h_val
                
            h_values.append(sum_of_h / (len(sequence) - d))
            hdiff_in_lengths = [10*(block / (len(sequence) - d)) for block in hdiff_in_lengths]
            hdiff_matrix.append(hdiff_in_lengths)
        return h_values, hdiff_matrix

    # Calculate H1
    H1_value, H1_matrix = calculate_h_values(sequence, H1)
    H2_value, H2_matrix = calculate_h_values(sequence, H2)
    H1_features = [10*i for i in H1_value]
    H2_features = [10*i for i in H2_value]

    # Calculate features
    w = 0.1  # Weight
    denominator = sum(H1_value + H2_value) * w + 1  # Constant denominator

    # Calculate APAAC descriptor
    apaac1 = [100 * w * h_value / denominator for h_value in H1_value]
    apaac2 = [100 * w * h_value / denominator for h_value in H2_value]
    
    return H1_features, H1_matrix, H2_features, H2_matrix, apaac1, apaac2

# Hydrophobicity list H1 (by LogP)
H1 = np.array([-0.512233954, -0.38383707, 1.710733781, -0.814662758])

# Hydrophilicity list H2 (by polar surface area)
H2 = np.array([-0.959054378, 1.359590767, -0.959054378, 0.558517989])

# Extract file data
proteinA = data['proteinA']
proteinB = data['proteinB']
sequenceA = data['Sequence A']
sequenceB = data['Sequence B']
interaction = data['interaction']

# Prepare feature lists
# sequenceA
proportionA_list=[]
lengthA_list=[]
one_mer_counts_listA_list=[]
one_mer_positions_listA_list=[]
two_mer_counts_listA_list=[]
two_mer_positions_listA_list=[]
three_mer_counts_listA_list=[]
three_mer_positions_listA_list=[]
H1_featuresA_list=[]
H2_featuresA_list=[]
H1_matrixA_list=[]
H2_matrixA_list=[]
apaac1A_list=[]
apaac2A_list=[]

# sequenceB
proportionB_list=[]
lengthB_list=[]
one_mer_counts_listB_list=[]
one_mer_positions_listB_list=[]
two_mer_counts_listB_list=[]
two_mer_positions_listB_list=[]
three_mer_counts_listB_list=[]
three_mer_positions_listB_list=[]
H1_featuresB_list=[]
H2_featuresB_list=[]
H1_matrixB_list=[]
H2_matrixB_list=[]
apaac1B_list=[]
apaac2B_list=[]

# Convert all lists to numpy arrays
interaction = np.array(interaction)
proteinA = np.array(proteinA, dtype='S')  # convert to byte string
proteinB = np.array(proteinB, dtype='S')  # convert to byte string

with h5py.File(output_file, 'w') as f:
    # Create datasets for featureS
    f.create_dataset('interaction', data=interaction)
    f.create_dataset('proteinA', data=proteinA)
    f.create_dataset('proteinB', data=proteinB)

# Extract features
print('Extracting proportionA,lengthA, one/two_mer_counts_listA, from protein A sequeces...')
for lineA in tqdm(sequenceA):
    # Call functions
    proportionA,lengthA = calculate_amino_acid_proportion(lineA)
    one_mer_counts_listA = calculate_nmer_position_percentage(lineA,1)
    two_mer_counts_listA = calculate_nmer_position_percentage(lineA,2)

    # Append to list
    proportionA_list.append(proportionA)
    lengthA_list.append(lengthA)
    one_mer_counts_listA_list.append(one_mer_counts_listA)
    two_mer_counts_listA_list.append(two_mer_counts_listA)

proportion_A = np.array(proportionA_list)
length_A = np.array(lengthA_list)
one_mer_countsA = np.array(one_mer_counts_listA_list)
two_mer_countsA = np.array(two_mer_counts_listA_list)

print(np.shape(proportion_A))
print(np.shape(length_A))
print(np.shape(one_mer_countsA))
print(np.shape(two_mer_countsA))

with h5py.File(output_file, 'a') as f:
    f.create_dataset('proportion_A', data=proportion_A)
    f.create_dataset('length_A', data=length_A)
    f.create_dataset('one_mer_countsA', data=one_mer_countsA)
    f.create_dataset('two_mer_countsA', data=two_mer_countsA)

proportion_A = []
length_A = []
one_mer_countsA = []
two_mer_countsA = []

print('Extracting three_mer_counts_listA, from protein A sequeces...')
for lineA in tqdm(sequenceA):
    # Call functions
    three_mer_counts_listA = calculate_nmer_position_percentage(lineA,3)
    
    # Append to list
    three_mer_counts_listA_list.append(three_mer_counts_listA)

three_mer_countsA = np.array(three_mer_counts_listA_list)
print(np.shape(three_mer_countsA))

with h5py.File(output_file, 'a') as f:
    f.create_dataset('three_mer_countsA', data=three_mer_countsA)

three_mer_countsA = []

print('Extracting proportionB,lengthB, one/two_mer_counts_listB, from protein B sequeces...')
for lineB in tqdm(sequenceB):
    # Call functions
    proportionB,lengthB = calculate_amino_acid_proportion(lineB)
    one_mer_counts_listB = calculate_nmer_position_percentage(lineB,1)
    two_mer_counts_listB = calculate_nmer_position_percentage(lineB,2)

    # Append to list
    proportionB_list.append(proportionB)
    lengthB_list.append(lengthB)
    one_mer_counts_listB_list.append(one_mer_counts_listB)
    two_mer_counts_listB_list.append(two_mer_counts_listB)

proportion_B = np.array(proportionB_list)
length_B = np.array(lengthB_list)
one_mer_countsB = np.array(one_mer_counts_listB_list)
two_mer_countsB = np.array(two_mer_counts_listB_list)

print(np.shape(proportion_B))
print(np.shape(length_B))
print(np.shape(one_mer_countsB))
print(np.shape(two_mer_countsB))

with h5py.File(output_file, 'a') as f:
    f.create_dataset('proportion_B', data=proportion_B)
    f.create_dataset('length_B', data=length_B)
    f.create_dataset('one_mer_countsB', data=one_mer_countsB)
    f.create_dataset('two_mer_countsB', data=two_mer_countsB)

proportion_B = []
length_B = []
one_mer_countsB = []
two_mer_countsB = []

print('Extracting three_mer_counts_listB, from protein B sequeces...')
for lineB in tqdm(sequenceB):
    # Call functions
    three_mer_counts_listB = calculate_nmer_position_percentage(lineB,3)
    
    # Append to list
    three_mer_counts_listB_list.append(three_mer_counts_listB)

three_mer_countsB = np.array(three_mer_counts_listB_list)
print(np.shape(three_mer_countsB))

with h5py.File(output_file, 'a') as f:
    f.create_dataset('three_mer_countsB', data=three_mer_countsB)

three_mer_countsB = []


input("Extraction complete. Press any key to exit...")

