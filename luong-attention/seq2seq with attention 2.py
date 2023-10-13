import pandas as pd
import numpy as np
import sys

from keras.layers import Input, LSTM, Dense, concatenate, Attention
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential

NUM_SAMPLES = 3000  # select 3000 samples
batch_size = 64
epochs = 2
latent_dim = 256  # the dim of status H of LSTM

data_path='cmn-eng/cmn.txt'
df=pd.read_table(data_path,header=None).iloc[ :NUM_SAMPLES,0:2] # 3000 x 2
print(df.head(5))
print()

df.columns=['inputs','targets']
df['targets']=df['targets'].apply(lambda x:'\t'+x+'\n') # Chinese, start sign: ‘\t’, stop sign:‘\n’
print(df.head(5))

input_texts = df.inputs.values.tolist()
target_texts = df.targets.values.tolist()

temp1 = df.inputs.unique() # 3000 sentences -> 2708 sentences, remove the duplicated sentences
temp2 = temp1.sum() # 2708 sentences -> 1 long string
temp3 = set(temp2) # a set with 67 unique chars
temp4 = list(temp3) # a list with 67 values
temp5 = sorted(temp4) # sorted

input_characters = sorted( list( set( df.inputs.unique().sum() ) ) ) # 3000 English sentences -> 67 unique chars
target_characters = sorted( list( set( df.targets.unique().sum() ) ) ) # 3000 Chinese sentences -> 1450 unique words


num_encoder_tokens = len(input_characters) # 67 tokens for English chars
num_decoder_tokens = len(target_characters) # 1450 tokens for Chinese words

INUPT_LENGTH = max([len(txt) for txt in input_texts]) # 18 English chars，一个样本最多包含 18 个字符
OUTPUT_LENGTH = max([len(txt) for txt in target_texts]) # 16 Chinese words，一个样本最多包含 16 个汉字

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)]) # English Char -> ID，字典：根据字母找ID
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)]) # Chinese Word -> ID，字典：根据汉字找ID

reverse_input_char_index = dict([(i, char) for i, char in enumerate(input_characters)]) # ID -> English Char，字典：根据ID找字母
reverse_target_char_index = dict([(i, char) for i, char in enumerate(target_characters)]) # ID -> Chinese Word，字典：根据ID找汉字
###################################################################################################

#                                   3000            18            67
encoder_input_data = np.zeros((NUM_SAMPLES, INUPT_LENGTH, num_encoder_tokens))   # all 0, 3000 sentences x 18 ENG chars x   67 one-hot encoding

#                                   3000            16            1450
decoder_input_data = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, num_decoder_tokens))  # all 0, 3000 sentences x 16 CHN words x 1450 one-hot encoding
decoder_target_data = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, num_decoder_tokens)) # all 0, 3000 sentences x 16 CHN words x 1450 one-hot encoding

print( input_token_index[" "]) # ID of "" in English is 0
print( target_token_index[" "]) # ID of "" in Chinese is 2


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # "Hi.", 3 English chars ---->  "\t嗨。\n", 4 Chinese words

    for t, char in enumerate(input_text):  # "Hi.", totally 3 English chars
        temp1 = input_token_index[char]
        encoder_input_data[i, t, temp1] = 1.0
    encoder_input_data[i, t + 1:, input_token_index[" "]] = 1.0  # the remaining chars in the sentence are all “ ”

    for t, word in enumerate(target_text):  # "\t嗨。\n", totally 4 Chinese words
        temp2 = target_token_index[word]
        decoder_input_data[i, t, temp2] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, temp2] = 1.0
    decoder_input_data[i, t + 1:   , target_token_index[" "]] = 1.0  # the remaining chars in the sentence are all “ ”
    decoder_target_data[i, t:   , target_token_index[" "]] = 1.0     # the remaining chars in the sentence are all “ ”

def MyPrint( mylist, row, col ):
    for x in range(row):
        for y in range(col):
            print( int(mylist[x,y]),end='')
        print(end="\n")
    return


print("------------ Sample 0: English Input, 18 x 67")
MyPrint(encoder_input_data[0], 18,67)
print("------------ Sample 0: Chinese Input, 16 x 1450")
MyPrint(decoder_input_data[0], 16,1450)
print("------------ Sample 0: Chinese Output, 16 x 1450")
MyPrint(decoder_target_data[0], 16,1450)


class Encoder(Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="encode_lstm")
    def call(self, inputs):
        encoder_outputs, state_h, state_c = self.encoder_lstm(inputs)
                       # Input: S x EL x 67;  S - Sample Number, EL - Encoder Input Sequence Length (18)
                       # Weight: (67 + 256 + 1) x 256 x 4 = 311766
                       # Output: S x EL x 256， S x 256, S x 256
        return encoder_outputs, state_h, state_c

class Decoder(Model):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decode_lstm")
        self.attention = Attention()  # default mode == "dot"

    def call(self, enc_outputs, dec_inputs, states_inputs):
        dec_outputs, dec_state_h, dec_state_c = self.decoder_lstm(dec_inputs, initial_state=states_inputs)
                       # Input: S x DL x 1450,  S x 256, S x 256; S - Sample Number, DL - Decoder Input Sequence Length (16)
                       # Weight: (1450 + 256 + 1) x 256 x 4 = 1747968
                       # Output: S x DL x 256， S x 256, S x 256
        attention_output = self.attention( [dec_outputs, enc_outputs] )
                       # Input: S x DL x 256,  S x EL x 256;
                       # no weights with the 'dot' model
                       # Output: S x DL x 256
                       # EL, Encoder Input Sequence Length, 18;
                       # DL, Decoder Input Sequence Length, 16 for training and 1 for inference

        return attention_output, dec_state_h, dec_state_c


def create_model(latent_dim):
    # Input Layer for Encoder, no weights
    encoder_inputs = Input( shape=(None, num_encoder_tokens) , name="encode_input")
                           # S x 18 x 67 for both training and inference

    # Encoder Layer
    encoder = Encoder(latent_dim) # 256
    encoder_outputs, encoder_final_state_h, encoder_final_state_c = encoder(encoder_inputs) # KerasTensor
    encoder_states_outputs = [encoder_final_state_h, encoder_final_state_c]  # [ S x 256, S x 256]

    # Input Layer for Decoder, no weights
    decoder_inputs = Input( shape=(None, num_decoder_tokens) , name="decode_input")
                           # S x 16 x 1450 for training and 1 x 1 x 1450 for inference

    # Decoder Layer
    decoder = Decoder(latent_dim) # 256
    attention_output, _ , _ = decoder(encoder_outputs, decoder_inputs, encoder_states_outputs) # 处理整个序列，只需要初始状态
    # Input: S x 18 x 256, S x 16 x 1450, [ S x 256, S x 256 ]
    # Output: S x 16 x 256, S x 256, S x 256

    # Dense Layer
    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="dense")
    dense_outputs = decoder_dense(attention_output)
                # Input: S x 16 x 256
                # Output: S x 16 x 1450

    ########## Training: seq2seq model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
                # Input: S x 18 x 67, S x 16 x 1450
                # Output: S x 16 x 1450

    ########## Inference: encoder model
    encoder_model = Model( inputs=encoder_inputs, outputs=[encoder_outputs, encoder_final_state_h, encoder_final_state_c])
            # Input: 1 x 18 x 67
            # Output: 1 x 18 x 256， 1 x 256, 1 x 256
            # 与model共享了一些层（参数），但有自己的Input和Output

    ########## Inference: decoder model # 只能处理1个词，因此每次都要输入状态
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, decoder_temp_state_h, decoder_temp_state_c = \
        decoder(encoder_outputs, decoder_inputs, decoder_state_inputs)   # 只能处理1个词，因此每次都要输入状态

    decoder_temp_states = [decoder_temp_state_h, decoder_temp_state_c]

    dense_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(inputs=[encoder_outputs, decoder_inputs, decoder_state_inputs],
                          outputs=[dense_outputs] + decoder_temp_states)     # [1] + [2 , 3] = [1,2,3]
             # Input: 1 x 18 x 256， 1 x 1 x 1450, [1 x 256, 1 x 256]
             # Output: 1 x 1 x 1450， 1 x 256, 1 x 256
             # 与model共享了一些层（参数），但有自己的Input和Output
    return model, encoder_model, decoder_model


model,encoder_model,decoder_model = create_model(latent_dim)

print(model.summary())
print(encoder_model.summary())
print(decoder_model.summary())


model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
epochs = 100
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs, validation_split=0.2)



def decode_sequence(input_seq, encoder_model, decoder_model):

    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq) # get the initial C and H by calling the encoder model
    states_value = [state_h, state_c]

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.  # generate the "\t",  the <START>

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:

        #MyPrint(target_seq[0], 1, 1450)

        temp1 = [encoder_outputs, [target_seq] , states_value]

        output_tokens, h, c = decoder_model.predict( temp1 ) # get the Output, H and C by calling the decoder model
        # print(output_tokens),  1450 probability values

        # get the output Chinese word
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit when <END> or full
        if sampled_char == '\n' or len(decoded_sentence) > INUPT_LENGTH:
            stop_condition = True

        # 更新target_seq
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update H and C
        states_value = [h, c]

    return decoded_sentence


samples = ["How are you!","Who are you?","Try it.","Tell me."]
for ss in samples:
    input_seq = np.zeros((1, INUPT_LENGTH, num_encoder_tokens))
    for t, char in enumerate(ss):
        input_seq[0, t, input_token_index[char]] = 1.0
    input_seq[0, t + 1:, input_token_index[" "]] = 1.0

    #MyPrint(input_seq[0], 18, 67)

    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
    print(ss)
    print('Decoded sentence:', decoded_sentence)
