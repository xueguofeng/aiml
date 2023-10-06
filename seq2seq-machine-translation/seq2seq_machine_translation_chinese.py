import pandas as pd
import numpy as np
import sys

from keras.layers import Input, LSTM, Dense, merge, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential

NUM_SAMPLES = 3000  # select 3000 samples
batch_size = 64
epochs = 100
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
MyPrint(encoder_input_data[0], 18,67)    # 3 x 67
print("------------ Sample 0: Chinese Input, 16 x 1450")
MyPrint(decoder_input_data[0], 16,1450)  # 4 x 1450
print("------------ Sample 0: Chinese Output, 16 x 1450")
MyPrint(decoder_target_data[0], 16,1450) # 3 x1450

def create_model():

    ####################################### For Training
    encoder_inputs = Input( shape=(None, num_encoder_tokens) )  # 67 tokens
    encoder = LSTM(latent_dim, return_state=True) # return_sequenes=False by default
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_state = [state_h, state_c]  # H of LSTM, C of LSTM - Conveyor Belt, O of LSTM - output is ignored
                          # the return_sequences is set to False by default
                          # 18 x 67 values -> [ 256 values, 256 values ], the state_O is ignored

    decoder_inputs = Input( shape=(None, num_decoder_tokens) )  # 1450 tokens
    decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True) # [ 256 values, 256 values ] as the initial state of C and H
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
                          # the return_sequences is set to True
                          # 1450 values -> 256 values

    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
                          # 256 values -> 1450 probability values

    # Training Model: [ 18 x 67 values, 16 x 1450 values ]  -> 16 x 1450 probability values
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    ####################################### For Inference - the Encoder part
    encoder_model = Model(encoder_inputs, encoder_state)
                          # share the encoder LSTM: all weights, intputs and outputs keep changing
                          # 18 x 67 values -> [ 256 values, 256 values ]

    ####################################### For Inference - the Decoder Part
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
                          # share the decoder LSTM and Softmax: all weights, intputs and outputs keep changing
                          # 1450 values -> 1450 probability values


#   plot_model(model=model, show_shapes=True)
#   plot_model(model=encoder_model, show_shapes=True)
#   plot_model(model=decoder_model, show_shapes=True)
    return model, encoder_model, decoder_model


def decode_sequence(input_seq, encoder_model, decoder_model):

    states_value = encoder_model.predict(input_seq) # get the initial C and H by calling the encoder model

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.  # generate the "\t",  the <START>

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:

        MyPrint(target_seq[0], 1, 1450)
        temp1 = [target_seq] + states_value
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


def train():
    model, encoder_model, decoder_model = create_model()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # 3 models share several some layers
    model.summary()
    encoder_model.summary()
    decoder_model.summary()

    # Train the 1 model, and other 2 models are updated as well
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    model.save('cmn-eng/s2s_chinese.h5')
    encoder_model.save('cmn-eng/encoder_model.h5')
    decoder_model.save('cmn-eng/decoder_model.h5')


def test():
    encoder_model = load_model('cmn-eng/encoder_model.h5', compile=False)
    decoder_model = load_model('cmn-eng/decoder_model.h5', compile=False)
    #encoder_model.summary()
    #decoder_model.summary()

    ss = input("Please input the English, 请输入要翻译的英文:")
    if ss == '-1':
        sys.exit()
    input_seq = np.zeros((1, INUPT_LENGTH, num_encoder_tokens))
    for t, char in enumerate(ss):
        input_seq[0, t, input_token_index[char]] = 1.0
    input_seq[0, t + 1:, input_token_index[" "]] = 1.0

    MyPrint(input_seq[0], 18, 67)

    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
    print('-')
    print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
#   intro = input("select train model or test model:")
#    intro = "test"
    intro = "train"

    if intro == "train":
        print("Training...........")
        train()
    else:
        print("testing.........")
        while (True):
            test()
