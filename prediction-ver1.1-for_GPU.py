from tensorflow.python.ops.gen_array_ops import prevent_gradient
import numpy as np
import os
import gc
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, Lambda
from tensorflow.keras import backend as K
from itertools import product
import h5py


# For GPU usage, set batch_size
batch_size = 2048 
RHP_features_route = '/content/drive/My Drive/RHP_deeplearning/RHP_features/RHP5020/DP500/extracted_features.hdf5'
fasta_file_route = '/content/drive/My Drive/RHP_deeplearning/protein_fasta'
model_route = '/content/drive/My Drive/RHP_deeplearning/PRI_prediction.h5'



# Define a translation dictionary
translation_dict = {
    "C": 1, "Y": 1, "A": 1, "T": 1, "G": 1,
    "S": 2, "Q": 2, "H": 2, "N": 2, "P": 2,
    "L": 3, "I": 3, "F": 3, "W": 3, "V": 3, "M": 3,
    "E": 4, "D": 4, "R": 4, "K": 4,
    "X": 5,  # Assign a special code for "X"
}

def translate_sequence(seq):
    # Translate the sequence
    num_list = [translation_dict.get(aa, 5) for aa in seq]

    # Convert the list to a string and remove unwanted characters
    num_str = str(num_list).replace("[", "").replace("]", "").replace(",", "").replace(" ", "")

    return num_str

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
    positions_list = []
    def count_n_mers(sequence,mer):
        counts=[0]*5
        start = 0
        position=[]
        while start < len(sequence):
            start_index = sequence.find(mer, start) # Locate and record the position of the first letter of n-mer
            if start_index == -1:  # If the n-mer is not found, end the loop
                break
            end_index = start_index + len(mer) - 1  # Calculate the position of the last letter of n-mer
            start_position = start_index / len(sequence)
            if n > 1:
                position.append(start_position)
            start_percentage = int(5 * start_position)  # Calculate the percentage of these mer's first and last letters relative to the length of the sequence
            end_percentage = int(5 * end_index / len(sequence))
            if start_percentage == end_percentage:
                counts[start_percentage] += 1
            for i in range(start_percentage, end_percentage):
                counts[i] += 1
            start = start_index + 1  # Update the starting position to find the next subsequence in the next iteration
        total = sum(counts) # Normalization
        if total > 0:  # To avoid division by zero
            for i, count in enumerate(counts):
                counts[i] = 10 * count / total
        return counts, position
    # Create a list to store the counts for each n-mer
    for n_mer in n_mer_list:
        con, pos = count_n_mers(sequence, n_mer)
        counts_list.append(con)
        if len(pos) < 128:
            pos.extend([0]*(128-len(pos)))  # Extend the list with zeros if its length is less than 1000
        else:
            pos = pos[:128]  # Truncate the list if its length is more than 1000
        positions_list.append(pos)
    return counts_list, positions_list

def Quasi_Order(sequence, maxlag):
    sequence = str(sequence)
    def tau_d(sequence, maxlag):
        tau_values = []
        sequence = list(map(int, str(sequence)))
        for d in range(1, maxlag+1):
            sum_of_squares = 0
            blur = 0
            item = 0
            for i in range(len(sequence) - d):
                # If 5 is present in the sequence, the chemical distance of the previous pair of amino acids is used
                if sequence[i] == 5 or sequence[i+d] == 5:
                    blur += 1
                else:
                    dis = chemical_distance_matrix[sequence[i]-1][sequence[i+d]-1]
                    item += 1
                    sum_of_squares += dis**2     
            sum_of_squares = blur*(sum_of_squares/item) + sum_of_squares
            tau_values.append(sum_of_squares)

        # Calculate the average of tau values
        mean_tau = sum(tau_values) / len(tau_values)
        # Computes the tau_values_feature, where each element is the corresponding element in tau_values minus the average value
        tau_values_feature = [val - mean_tau for val in tau_values]
        # Normalization
        max_val = max(tau_values_feature)
        min_val = min(tau_values_feature)
        tau_values_feature = [(val - min_val) / (max_val - min_val) for val in tau_values_feature]
        return tau_values, tau_values_feature

    # Calculated tau_values and tau_values_feature
    tau_values, tau_values_feature = tau_d(sequence, maxlag)
    
    one_mer_frequencies = calculate_frequencies(sequence,get_all_n_mers(1))
    two_mer_frequencies = calculate_frequencies(sequence,get_all_n_mers(2))

    # Calculae quasi features
    w = 0.0001  # weight
    denominator = sum(tau_values) * w + 1  # Constant denominator can be calculated at first
    one_mer_features = [fr / denominator for fr in one_mer_frequencies]
    two_mer_features = [fr / (denominator*0.1) for fr in two_mer_frequencies]

    # Calculate Quasi-Order type II
    quasi_type2 = [w * tau_value / (denominator*0.1) for tau_value in tau_values]

    return tau_values_feature, one_mer_features, two_mer_features, quasi_type2

def APAAC(sequence, maxlag):
    sequence = str(sequence)
    def calculate_h_values(sequence, H):
        sequence = list(map(int, str(sequence)))
        h_values = []
        for d in range(1, maxlag+1):
            sum_of_product = 0
            blur = 0
            item = 0
            for i in range(len(sequence) - d):
                # If 5 is present in the sequence, the h value result of the previous pair of amino acids is used
                if sequence[i] == 5 or sequence[i+d] == 5:
                    blur += 1
                else:
                    h_val = H[sequence[i]-1] * H[sequence[i+d]-1]
                    item += 1
                    sum_of_product += h_val
            sum_of_product = blur*(sum_of_product/item) + sum_of_product
            h_values.append(sum_of_product / (len(sequence) - d))
        return h_values

    # Calculate H1 and H2
    H1_value = calculate_h_values(sequence, H1)
    H2_value = calculate_h_values(sequence, H2)
    H1_features = [10*i for i in H1_value]
    H2_features = [10*i for i in H2_value]
    # Get all 1-mer and 2-mer
    all_one_mers = get_all_n_mers(1)
    all_two_mers = get_all_n_mers(2)
    # Calculate the frequencies of all 1-mer and 2-mer in the sequence
    one_mer_frequencies = calculate_frequencies(sequence, all_one_mers)
    two_mer_frequencies = calculate_frequencies(sequence, all_two_mers)

    # Calculate features
    w = 0.1  # Weight
    denominator1 = sum(H1_value) * w + 1  # Constant denominator
    denominator2 = sum(H2_value) * w + 1  # Constant denominator
    denominator3 = sum(H1_value + H2_value) * w + 1  # Constant denominator

    # Calculate APAAC type I
    apaac_type1_one_mer=[fr / denominator3 for fr in one_mer_frequencies]
    apaac_type1_two_mer=[10 * fr / denominator3 for fr in two_mer_frequencies]

    # Calculate APAAC type II
    apaac_type2_1 = [100 * w * h_value / denominator3 for h_value in H1_value]
    apaac_type2_2 = [100 * w * h_value / denominator3 for h_value in H2_value]
    
    return H1_features, H2_features, apaac_type1_one_mer, apaac_type1_two_mer, apaac_type2_1, apaac_type2_2

# Define Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def get_positional_encoding(seq_length, d_model):
    # Initialize a matrix of zeros with shape (seq_length, d_model)
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((seq_length, d_model))
    # Apply the positional encoding formula
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    pos_enc = pos_enc[np.newaxis, ...]
    return tf.cast(pos_enc, dtype=tf.float32)

def add_positional_encoding(inputs):
    _, seq_length, d_model = inputs.shape.as_list()
    pos_enc = get_positional_encoding(seq_length, d_model)
    return inputs + pos_enc

# Apply Transformer Layer to process_A and process_B functions
def process_A(input_A):
    out_A = Dense(1024, activation='relu')(input_A)
    out_A = Dense(1024, activation='relu')(out_A)
    out_A = Dense(512, activation='relu')(out_A)
    return out_A

def process_B(input_B):
    out_B = Dense(1024, activation='relu')(input_B)
    out_B = Dense(1024, activation='relu')(out_B)
    out_B = Dense(512, activation='relu')(out_B)
    return out_B

def stack_tensors(tensors):
    return K.stack(tensors, axis=1)

def create_model(input_shape_A, input_shape_B):
    # Define the input layers
    input_A = Input(shape=(input_shape_A,))
    input_B = Input(shape=(input_shape_B,))
    # Process inputs
    out_A = process_A(input_A)
    out_B = process_B(input_B)

    stacked = Lambda(stack_tensors)([out_A, out_B])
    pos_stacked = Lambda(add_positional_encoding)(stacked)
    transformer_block1 = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=32)
    out = transformer_block1(pos_stacked)

    out = Flatten()(out)
    
    # Further processing
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.1)(out)
    out = Dense(256, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = Dense(128, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)

    # Compile model
    model = Model(inputs=[input_A, input_B], outputs=out)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# chemical distance matrix
chemical_distance_matrix = np.array([
    [0, 0.884954128, 0.682926829, 0.684210526],
    [0.884954128, 0, 0.838060351, 0.821937322],
    [0.682926829, 0.838060351, 0, 0.653846154],
    [0.684210526, 0.821937322, 0.653846154, 0]
])

# Hydrophobicity list H1 (by LogP)
H1 = np.array([-0.512233954, -0.38383707, 1.710733781, -0.814662758])

# Hydrophilicity list H2 (by polar surface area)
H2 = np.array([-0.959054378, 1.359590767, -0.959054378, 0.558517989])

# Load the model
model = load_model(
    model_route, 
    custom_objects={
    'TransformerBlock': TransformerBlock, 
    'stack_tensors': stack_tensors,
    'add_positional_encoding': add_positional_encoding,
    'get_positional_encoding': get_positional_encoding
    }
)

# Reads the list of all fasta file names in the current folder
# Gets the directory where the current python script is located
fasta_dir = fasta_file_route
fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]

# Initializes an empty DataFrame to store all predictions
all_predictions = pd.DataFrame()

# Open the HDF5 file
with h5py.File(RHP_features_route, 'r') as f:
    # Read datasets
    one_mer_countsA = f['one_mer_countsA'][:]
    two_mer_countsA = f['two_mer_countsA'][:]
    three_mer_countsA = f['three_mer_countsA'][:]
    proportion_A = f['proportion_A'][:]
    length_A = f['length_A'][:]
    tau_values_feature_A = f['tau_values_feature_A'][:]
    one_mer_features_A = f['one_mer_features_A'][:]
    two_mer_features_A = f['two_mer_features_A'][:]
    quasi_type2A = f['quasi_type2A'][:]
    H1_featuresA = f['H1_featuresA'][:]
    H2_featuresA = f['H2_featuresA'][:]
    apaac_type1_one_merA = f['apaac_type1_one_merA'][:]
    apaac_type1_two_merA = f['apaac_type1_two_merA'][:]
    apaac_type2_1A = f['apaac_type2_1A'][:]
    apaac_type2_2A = f['apaac_type2_2A'][:]

    # Flattening mers data
    one_mer_countsA_flattened = one_mer_countsA.reshape(one_mer_countsA.shape[0], -1)
    two_mer_countsA_flattened = two_mer_countsA.reshape(two_mer_countsA.shape[0], -1)
    three_mer_countsA_flattened = three_mer_countsA.reshape(three_mer_countsA.shape[0], -1)
    # Put all the features A into a 1D list
    featuresA = [one_mer_countsA_flattened, two_mer_countsA_flattened, three_mer_countsA_flattened, proportion_A, tau_values_feature_A, one_mer_features_A, two_mer_features_A, quasi_type2A, H1_featuresA, H2_featuresA, apaac_type1_one_merA, apaac_type1_two_merA, apaac_type2_1A, apaac_type2_2A]
    # Reshape length
    length_A_reshaped = length_A.reshape(-1, 1)
    # Concatenate all features and length
    final_featuresA = np.concatenate(featuresA + [length_A_reshaped], axis=1)
    final_featuresA = np.array(final_featuresA)
    total_steps = final_featuresA.shape[0]

#Run the prediction program 

for fasta_file in fasta_files:
    # Read a fasta file in the list
    fasta_sequences = SeqIO.parse(open(os.path.join(fasta_dir, fasta_file)), 'fasta')
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)

    # RHPize protein sequence
    sequenceB = translate_sequence(sequence)
    print(calculate_amino_acid_proportion(sequenceB))

    # Extract features for protein sequence as sequence B
    proportionB,lengthB = calculate_amino_acid_proportion(sequenceB)
    one_mer_counts_listB, one_mer_positions_listB = calculate_nmer_position_percentage(sequenceB,1)
    two_mer_counts_listB, two_mer_positions_listB = calculate_nmer_position_percentage(sequenceB,2)
    three_mer_counts_listB, three_mer_positions_listB = calculate_nmer_position_percentage(sequenceB,3)
    #five_mer_counts_listB, five_mer_positions_listB = calculate_nmer_position_percentage(sequenceB,5)
    tau_values_featureB, one_mer_featuresB, two_mer_featuresB, quasi_type2B = Quasi_Order(sequenceB,30)
    H1_featuresB, H2_featuresB, apaac_type1_one_merB, apaac_type1_two_merB, apaac_type2_1B, apaac_type2_2B = APAAC(sequenceB,30)
    one_mer_counts_listB_flattened = [item for sublist in one_mer_counts_listB for item in sublist]
    two_mer_counts_listB_flattened = [item for sublist in two_mer_counts_listB for item in sublist]
    three_mer_counts_listB_flattened = [item for sublist in three_mer_counts_listB for item in sublist]

    #Get final_featuresB
    final_featuresB = np.array(one_mer_counts_listB_flattened + two_mer_counts_listB_flattened + three_mer_counts_listB_flattened + proportionB + tau_values_featureB + one_mer_featuresB + two_mer_featuresB + quasi_type2B + H1_featuresB + H2_featuresB + apaac_type1_one_merB + apaac_type1_two_merB + apaac_type2_1B + apaac_type2_2B + [lengthB])
    final_featuresB = final_featuresB.reshape(1, -1)
    proteinB_features = np.repeat(final_featuresB, batch_size, axis=0)

    # Prepare to save all predictions
    predictions = []

    # For each row of A (each protein), predictions are made against B
    for i in tqdm(range(0, total_steps, batch_size), desc=f"Processing protein: {name}"):

        if i + batch_size <= len(final_featuresA):
            # Not the last batch
            proteinA_features = final_featuresA[i:i+batch_size]
        else:
        # Last batch
            remainder = len(final_featuresA) - i
            proteinA_features = final_featuresA[-remainder:]
            proteinB_features = np.repeat(final_featuresB, remainder, axis=0)

        # Predict
        prediction = model.predict([proteinA_features, proteinB_features], verbose=0)

        # Save the prediction result to a list
        predictions.extend(prediction.tolist())

    # Change the predictions array to 1D list，save to DataFrame
    predictions_df = pd.DataFrame(np.array(predictions).flatten(), columns=[name])
    
    # Create a new dataframe, or splice the prediction behind an existing datafame
    if all_predictions.empty:
        all_predictions = predictions_df
    else:
        all_predictions = pd.concat([all_predictions, predictions_df], axis=1)
    # Manually trigger tensorflow garbage collection
    K.clear_session()

    # Reset cutted proteinB_features in the last batch
    proteinB_features = np.repeat(final_featuresB, batch_size, axis=0)

# Save all predictions to a CSV file
all_predictions.to_csv('all_predictions.csv', index=False)

print("Prediction complete. The prediction.csv has been generated. Press any key to exit...")
