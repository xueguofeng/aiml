import numpy as np
import tensorflow as tf
from tensorflow import keras

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.

data_path = "fra-eng/fra.txt"

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))  # 70 chars in English
target_characters = sorted(list(target_characters)) # 93 chars in French
num_encoder_tokens = len(input_characters) # English, 70 chars
num_decoder_tokens = len(target_characters) # French, 93 chars
max_encoder_seq_length = max([len(txt) for txt in input_texts]) # English sentence, max 14 chars
max_decoder_seq_length = max([len(txt) for txt in target_texts]) # French sentence, max 59 chars

print("Number of samples:", len(input_texts)) # 10000 English sentences, 10000 French sentences
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
###################################################################################################


#                                   10000            14                     70
encoder_input_data = np.zeros( (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

#                                   10000            59                      93
decoder_input_data = np.zeros( (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros( (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # input_test               target_text
    # "Go." ------------------ "\tVa!\n"

    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1:   , input_token_index[" "]] = 1.0    # the remaining chars in the sentence are all “ ”

    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    decoder_input_data[i, t + 1:   , target_token_index[" "]] = 1.0  # the remaining chars in the sentence are all “ ”
    decoder_target_data[i, t:   , target_token_index[" "]] = 1.0     # the remaining chars in the sentence are all “ ”

###################################################################################################
def MyPrint( mylist, row, col ):
    for x in range(row):
        for y in range(col):
            print( int(mylist[x,y]),end='')
        print(end="\n")
    return
###################################################################################################


# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("fra-eng/s2s_french.h5")
model.summary()


encoder_inputs = model.input[0]  # input_1, English, 14 x 70
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
                  # encoder lstm, includes Output(256 values), H(256 values), C(256 values)
encoder_states = [state_h_enc, state_c_enc]

encoder_model = keras.Model(encoder_inputs, encoder_states,name="encoder")
                     # model input:  English, 14 x 70
                     # model output:  [ H(256 values), C(256 values) ]
encoder_model.summary()

decoder_inputs = model.input[1]  # input_2, French, 59 x 93
decoder_state_input_h = keras.Input( shape=(latent_dim,), name = 'input_h' )
decoder_state_input_c = keras.Input( shape=(latent_dim,), name = 'input_c' )
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states, name="decoder")
decoder_model.summary()





# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.

    #MyPrint(input_seq[0], 14, 70)

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

        #MyPrint(target_seq[0], 1, 93)
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

for seq_index in range(0,20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("--------------------------------------------------")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

