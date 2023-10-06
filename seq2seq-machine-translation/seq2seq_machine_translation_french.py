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


print( input_token_index[" "]) # ID of "" in English is 0
print( target_token_index[" "]) # ID of "" in French is 2

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


print("------------ Sample 0: English Input, 14 x 70")
MyPrint(encoder_input_data[0], 14,70)
print("------------ Sample 0: French Input, 59 x 93")
MyPrint(decoder_input_data[0], 59,93)
print("------------ Sample 0: French Output, 59 x 93")
MyPrint(decoder_target_data[0], 59,93)
###################################################################################################


########## Define an input sequence and process it.
encoder_inputs = keras.Input( shape=(None, num_encoder_tokens) )
                              # the English input is N x 70, N means arbitrarily long; we will feed 14 x 70 for a sample
encoder = keras.layers.LSTM( latent_dim, return_state=True )
                              # H contains 256 values; C also contains 256 values by LSTM
                              # return_state is true, return_sequences is false by default
encoder_outputs, state_h, state_c = encoder( encoder_inputs )
# So, for each English sample - 14 x 70 values, we get an Output(256 values), a H(256 values) and a C(256 values)

encoder_states = [state_h, state_c] # We discard `encoder_outputs` and only keep the states.



########## Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input( shape=(None, num_decoder_tokens) )
                             # the French input is N x 93, N means arbitrarily long; we will feed 59 x 93 for a sample

# We set up our decoder to return full output sequences and to return internal states as well.
# We don't use the return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
                              # set the initial_state from the encoder
                              # return_sequences is set to True
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
# So, for each French sample - 59 x 93 values, we get an 59 x 93(one-hot)


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("fra-eng/s2s_french.h5")