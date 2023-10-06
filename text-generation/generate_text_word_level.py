# keras module for building LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import tensorflow.keras.utils as ku

# set seeds for reproducability
from tensorflow import random
from numpy.random import seed
random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

curr_dir = 'archive/'
all_headlines = []
for filename in os.listdir(curr_dir):
    print(filename)
    if 'Articles' in filename:
        article_df = pd.read_csv(curr_dir + filename)
#       print(article_df)
        temp = list(article_df.headline.values)
        all_headlines.extend( temp )
        break

all_headlines = [h for h in all_headlines if h != "Unknown"] # 886 headlines to 831 headlines
print(len(all_headlines))


def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [ clean_text(x) for x in all_headlines ] # remove punctuations in the headlines and lower-case all words
print(corpus[0:5])

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization

    tokenizer.fit_on_texts(corpus)                   # ID:1~2421

    total_words = len(tokenizer.word_index) + 1      # ID:0, reserved
                                                     # so the len is set to 2422

    ## convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        #token_list = tokenizer.texts_to_sequences([line])[0]
        temp = tokenizer.texts_to_sequences([line])  # word -> ID, sequence of words -> sequence of IDs
        token_list = temp[0] # [[x,y,z...]] -> [x,y,z...]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words


inp_sequences, total_words = get_sequence_of_tokens(corpus) # 831 headlines
print(inp_sequences[:5])                                    # 4806 ngram phrases and 2422 unique words


def generate_padded_sequences(input_sequences):
    max_sequence_len = max( [len(x) for x in input_sequences] ) # the longest sequence of tokens: 19
    input_sequences = np.array( pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre') )

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1] # input 18, output 1, totally 19
    # predictors, 18 IDs: 0 0 0 0 1~2241
    # label, 1 ID: 1~2241

    label = ku.to_categorical(label, num_classes=total_words) # 4806 labels for 4806 ngram phrases, a vector with 2422 values for each label
                                                              # use one-hot encoding(2242) to represent ID(1~2421)
    return predictors, label, max_sequence_len
    # predictors, 4806 ngram phrases x [ 18 IDs ]
    # label, 4806 labels x [ 2422 one-hot encoding ]

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    #                        2242      10       18
    model.add( Embedding( total_words, 10, input_length=input_len) )
    # totally 2422 words and every word vector has 10 components;
    # 1 input has 18 words
    # 2422 x 10 = 24,220 parameters

    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    # the status-H has 100 values
    # (10 W + 100 W + 1b) x 100 x 4 = 44400 parameters

    model.add(Dropout(0.1))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    # Softmax
    # Inputs: 100 values from the last State H
    # Outputs: 2422 values
    # Parameters: (100 W + 1 b) x 2422 = 244622

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


model = create_model(max_sequence_len, total_words)
model.summary()

model.fit(predictors, label, epochs=100, verbose=5)


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # [[x,y,z...]] -> [x,y,z...]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        #predicted = model.predict_classes(token_list, verbose=0)
        temp = model.predict(token_list) # 2422 possibility values
        predicted = np.argmax(temp,axis=1)
              # By default, the index is into the flattened array, otherwise along the specified axis.

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text.title()


print (generate_text("President", 10, model, max_sequence_len))
print (generate_text("Congress", 10, model, max_sequence_len))
print (generate_text("Food", 10, model, max_sequence_len))
print (generate_text("Mexican border", 10, model, max_sequence_len))

