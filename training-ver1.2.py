import pandas as pd
import numpy as np
import h5py
import tensorflow as tf
from itertools import product
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Open the HDF5 file
with h5py.File('extracted_features.hdf5', 'r') as f:
    # Read datasets
    interactions = f['interaction'][:]
    proteinA = f['proteinA'][:]
    proteinB = f['proteinB'][:]
    
    one_mer_countsA = f['one_mer_countsA'][:]
    two_mer_countsA = f['two_mer_countsA'][:]
    three_mer_countsA = f['three_mer_countsA'][:]
    one_mer_countsB = f['one_mer_countsB'][:]
    two_mer_countsB = f['two_mer_countsB'][:]
    three_mer_countsB = f['three_mer_countsB'][:]

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

    proportion_B = f['proportion_B'][:]
    length_B = f['length_B'][:]
    tau_values_feature_B = f['tau_values_feature_B'][:]
    one_mer_features_B = f['one_mer_features_B'][:]
    two_mer_features_B = f['two_mer_features_B'][:]
    quasi_type2B = f['quasi_type2B'][:]
    H1_featuresB = f['H1_featuresB'][:]
    H2_featuresB = f['H2_featuresB'][:]
    apaac_type1_one_merB = f['apaac_type1_one_merB'][:]
    apaac_type1_two_merB = f['apaac_type1_two_merB'][:]
    apaac_type2_1B = f['apaac_type2_1B'][:]
    apaac_type2_2B = f['apaac_type2_2B'][:]

    # Flattening mers data
    one_mer_countsA_flattened = one_mer_countsA.reshape(one_mer_countsA.shape[0], -1)
    two_mer_countsA_flattened = two_mer_countsA.reshape(two_mer_countsA.shape[0], -1)
    three_mer_countsA_flattened = three_mer_countsA.reshape(three_mer_countsA.shape[0], -1)
    one_mer_countsB_flattened = one_mer_countsB.reshape(one_mer_countsB.shape[0], -1)
    two_mer_countsB_flattened = two_mer_countsB.reshape(two_mer_countsB.shape[0], -1)
    three_mer_countsB_flattened = three_mer_countsB.reshape(three_mer_countsB.shape[0], -1)

    # Put all the features into a 1D list
    featuresA = [one_mer_countsA_flattened, two_mer_countsA_flattened, three_mer_countsA_flattened, proportion_A, tau_values_feature_A, one_mer_features_A, two_mer_features_A, quasi_type2A, H1_featuresA, H2_featuresA, apaac_type1_one_merA, apaac_type1_two_merA, apaac_type2_1A, apaac_type2_2A]
    featuresB = [one_mer_countsB_flattened, two_mer_countsB_flattened, three_mer_countsB_flattened, proportion_B, tau_values_feature_B, one_mer_features_B, two_mer_features_B, quasi_type2B, H1_featuresB, H2_featuresB, apaac_type1_one_merB, apaac_type1_two_merB, apaac_type2_1B, apaac_type2_2B]
    # Reshape length
    length_A_reshaped = length_A.reshape(-1, 1)
    length_B_reshaped = length_A.reshape(-1, 1)

    # Concatenate all features and length
    final_featuresA = np.concatenate(featuresA + [length_A_reshaped], axis=1)
    final_featuresB = np.concatenate(featuresB + [length_B_reshaped], axis=1)
    # Get an array from 0 to n-1, where n is the number of samples
    n_samples = final_featuresA.shape[0]
    indices = np.arange(n_samples)
    # Randomly sort an indexed array
    np.random.shuffle(indices)
    # Use this randomly sorted index array to sort the two feature arrays
    final_featuresA = final_featuresA[indices]
    final_featuresB = final_featuresB[indices]
    # Interactions, sorted by the same index array
    interactions = interactions[indices]

def split_data(featuresA, featuresB, interactions, train_ratio, validation_ratio):
    # Calculate the size of the training set
    train_size = int(len(featuresA) * train_ratio)
    # Calculate the size of the validation set
    validation_size = int(len(featuresA) * validation_ratio)
    # Split the training set
    featuresA_train = featuresA[:train_size]
    featuresB_train = featuresB[:train_size]
    interactions_train = interactions[:train_size]
    # Split the validation set
    featuresA_val = featuresA[train_size:train_size+validation_size]
    featuresB_val = featuresB[train_size:train_size+validation_size]
    interactions_val = interactions[train_size:train_size+validation_size]
    # Split the test set
    featuresA_test = featuresA[train_size+validation_size:]
    featuresB_test = featuresB[train_size+validation_size:]
    interactions_test = interactions[train_size+validation_size:]
    return featuresA_train, featuresB_train, interactions_train, featuresA_val, featuresB_val, interactions_val, featuresA_test, featuresB_test, interactions_test

# 定义训练集、验证集和测试集的比例
train_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2 # 剩余部分
#划分训练集和测试集
X_A_train, X_B_train, Y_train, X_A_val, X_B_val, Y_val, X_A_test, X_B_test, Y_test = split_data(final_featuresA, final_featuresB, interactions, train_ratio, validation_ratio)

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

# Define the shape (features amount) of your input here
input_shape_A = 1021
input_shape_B = 1021

# Create model
model = create_model(input_shape_A, input_shape_B)

# 训练模型
history = model.fit([X_A_train, X_B_train], Y_train, validation_data=([X_A_val, X_B_val], Y_val), epochs=10, batch_size=32)

# 绘制训练过程中的损失曲线
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# 绘制准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 使用模型进行预测并绘制ROC曲线
Y_pred = model.predict([X_A_test, X_B_test])
fpr, tpr, _ = roc_curve(Y_test, Y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
