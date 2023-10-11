# https://notebook.community/tensorflow/examples/community/en/nmt_with_luong_attention

import tensorflow as tf

import numpy as np
import unicodedata
import re

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


raw_data_en, raw_data_fr = list(zip(*raw_data)) # '*' is to unpack the tuple (20 values) first
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

raw_data_en = [normalize_string(data) for data in raw_data_en] # Encoder Input

raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr] # Decorder Input
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]  # Decorder Output

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)  # 97 English words; vectorize a text corpus and provide a ID for each word (0 is reserved)
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post') # 20 sentences x 10 English words

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)  # 109 French words
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in,padding='post')  # 20 sentences x 14 French words
data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,padding='post') # 20 sentences x 14 French words

BATCH_SIZE = 5 # batch size = 5, so number of batch = 4
dataset = tf.data.Dataset.from_tensor_slices( (data_en, data_fr_in, data_fr_out) )
dataset = dataset.shuffle(20).batch(BATCH_SIZE)
       # total 4 batches, 5 samples for each batch

       # for each sample:    Encoder Input     Decorder Input       Decorder Output
       #                    (10 English words, 14 French words) -> (14 French words)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size # 64
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size) # 98 x 32
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states): # one call can handle S x N, and the 1st call can use 1 x 1 for the initialization
        embed = self.embedding(sequence)     # Input: S x N, ID (0~97)
                # S and N will not influence the number of weights, which is decided by vocal_size and embedding_size
                                             # Output: S x N x 32, Word Vector
                                             # The word vector of ID 0 are not all 0s (32 zeros)
        output, state_h, state_c = self.lstm(embed, initial_state=states) # S x 2 initial states，H and C
                # Input: S x N x 32
                # S and N will not influence the number of weights,
                # but the embedding size (32) will do: (32 + 64 + 1) x 64 x 4
                                             # Output: S x N x 64
                                             # State-H: S x 64
                                             # State-C：S x 64

        #temp = output[0,9,:] # is equal to the final state_h
        #print(temp)
        #print(state_h)

        return output, state_h, state_c

    def init_states(self, batch_size):
        return ( tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size]) )

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size  # 64
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size) # 110 x 32
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size) # 110 neurons, the input and weights are not decided at this point

    def call(self, sequence, state):
        embed = self.embedding(sequence) # Input: S x N, ID (0~109)
                # S and N will not influence the number of weights, which is decided by vocal_size and embedding_size
                                         # Output: S x N x 32, Word Vector
                                         # The word vector of ID 0 are not all 0s (32 zeros)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
                  # Input: S x N x 32
                  # S and N will not influence the number of weights,
                  # but the embedding size (32) will do: (32 + 64 + 1) x 64 x 4
                                         # Output: S x N x 64
                                         # State-H: S x 64
                                         # State-C：S x 64
                                         # Notes: we can get S x N x 64 by 1 lstm call

        logits = self.dense(lstm_out)   # Input: S x N x 64 (EagerTensor)
                                        # 110 neurons
                                        # Weights: (64 + 1) x 110 = 7150, not relevant to S and N
                                        # no activation function
                                        # Output:  S x N x 110
        return logits, state_h, state_c

    
#######################################################
# Initialize the model (number of weights and initial values) by feeding in some dummy data (EagerTensor).

en_vocab_size = len(en_tokenizer.word_index) + 1 # the reserved 0 is for ' '
fr_vocab_size = len(fr_tokenizer.word_index) + 1 # the reserved 0 is for ' '

EMBEDDING_SIZE = 32
LSTM_SIZE = 64

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

initial_states = encoder.init_states(1) # the initial State-H (1x64) and State-C (1x64 ) for the encoder
# Initialize the model with the dummy data: 1 x 10 (2-D，a batch with 1 sample)
encoder_outputs = encoder(tf.constant( [ [1, 2, 3, 4, 5, 6, 7, 8, 9, 0] ] ), initial_states)
# Initialize the model with the dummy data: 1 x 14 (2-D，a batch with 1 sample)
decoder_outputs = decoder(tf.constant( [ [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]  ] ), encoder_outputs[1:])


def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
          # from_logits=True, the softmax will be applied
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
          # mask those zeros out when computing the loss and updating weights (for the decorder output)

    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam()


# @tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states) # 5x10, (5x64,5x64)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)
                  # Y:5x14 (some ID=0)    Y_hat: 5x14x110
                  # the loss from 5 samples
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
        # encoder: embedding layer, 98 x 32
        # encoder: LSTM layer, (32 + 64 + 1) x 64 x 4
        # decoder: embedding layer, 110 x 32
        # decoder: LSTM layer, (32 + 64 + 1) x 64 x 4
        # decoder: Dense - fully connected layer, (64 + 1) x 110
    return loss



def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    # print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)
                          # no padding

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []

    while True:
        de_output, de_state_h, de_state_c = decoder(de_input, (de_state_h, de_state_c))
                        # update the state-h and state-c

        temp1 = tf.argmax(de_output, -1) # 1x 110 probabilities -> 1 ID

        de_input = tf.argmax(de_output, -1) # the next input will be the current output

        temp2 = de_input.numpy()
        temp3 = de_input.numpy()[0][0]

        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))

##################### Train the model
NUM_EPOCHS = 300
for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    # Get 5 initial encoder State-H and State-C,  ( 5 x 64, 5 x 64 )
    # Samples are independent to each other, and all samples are handled with the same Encoder LSTM initial states

#    predict()

    # the dataset includes 4 batches; take(-1) will get all the 4 batches
    temp1 = dataset.take(-1)
    temp2 = enumerate( temp1 )

    # batch: 0,1,2,3
    # each batch contains 5 samples
    # for each sample:    Encoder Input     Decorder Input       Decorder Output
    #                    (10 English words, 14 French words) -> (14 French words)
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,target_seq_out, en_initial_states)
                           # 5 samples and 5 initial encoder State-H and State-C
                           # 5 samples(English) are irrelevant to each other
                           # Calculate the loss from 1 batch with 5 samples
    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))



test_sents = (
    'What a ridiculous concept!',
    'Your idea is not entirely crazy.',
    "A man's worth lies in what he is.",
    'What he did is very wrong.',
    "All three of you need to do that.",
    "Are you giving me another chance?",
    "Both Tom and Mary work as models.",
    "Can I have a few minutes, please?",
    "Could you close the door, please?",
    "Did you plant pumpkins this year?",
    "Do you ever study in the library?",
    "Don't be deceived by appearances.",
    "Excuse me. Can you speak English?",
    "Few people know the true meaning.",
    "Germany produced many scientists.",
    "Guess whose birthday it is today.",
    "He acted like he owned the place.",
    "Honesty will pay in the long run.",
    "How do we know this isn't a trap?",
    "I can't believe you're giving up.",
)

for test_sent in test_sents:
    test_sequence = normalize_string(test_sent)
    predict(test_sequence)