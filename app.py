import tensorflow as tf

import numpy as np
from tensorflow import keras
import random
import string
import pandas as pd
import re
from tqdm import tqdm
import pickle

num_samples = 9000   # Number of samples to train on.
data_path = "kannada_corpus.csv"  # Path to the data txt file on disk.

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
output_dim = 64


# Assume lines is already read from the dataset file
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

# Load the dataset
data_path = 'kannada_corpus.csv'  # Update the path to your dataset
with open(data_path, 'r', encoding='utf-8') as file:
    lines = file.read().split('\n')

for line in lines:
    if not line.strip():
        continue
    input_text = line.strip()
    target_text = "\t" + input_text + "\n"  # Add start and end tokens for target texts

    # Introduce synthetic errors
    input_text = re.sub(r'[^ಅ-ಹಅ-ಹ ]+', '', input_text)  # Keep only Kannada characters and spaces
    input_text = ''.join(random.choice(input_text) if random.random() < 0.1 else char for char in input_text)

    input_texts.append(input_text.lower())
    target_texts.append(target_text)

    # Collect unique characters
    for char in input_text:
        input_characters.add(char)
    for char in target_text:
        target_characters.add(char)

# Ensure space character is included
input_characters.add(' ')
target_characters.add(' ')

# Tokenize the text
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# Prepare encoder and decoder input and output data
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.0  # Padding

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.0  # Padding
    decoder_target_data[i, t:, target_token_index[' ']] = 1.0  # Padding


input_token_index

model = keras.models.load_model('auto.h5')

# Define sampling models
# Restore the model and construct the encoder and decoder.
#model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

test_text = "ಕನಡ"

encoder_test_data = np.zeros(
    (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")

for t, char in enumerate(test_text):
    encoder_test_data[0, t, input_token_index[char]] = 1.0

decoded_sentence = decode_sequence(encoder_test_data)
print(decoded_sentence)