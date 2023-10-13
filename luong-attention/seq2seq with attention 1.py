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


raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

raw_data_en = [normalize_string(data) for data in raw_data_en]  # Encoder Input

raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]  # Decorder Input
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]  # Decorder Output

en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)  # 97 English words
data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')  # 20 sentences x 10 English words

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)  # 109 French words
data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')  # 20 sentences x 14 French words
data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out,
                                                            padding='post')  # 20 sentences x 14 French words

BATCH_SIZE = 5
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(BATCH_SIZE)


# total 4 batches, 5 samples for each batch

# for each sample:    Encoder Input     Decorder Input       Decorder Output
#                    (10 English words, 14 French words) -> (14 French words)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size  # 64
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)  # 98, 32
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence) # Input: S x N, ID (0~97); Output: S x N x 32, Word Vector
        output, state_h, state_c = self.lstm(embed, initial_state=states) # Input: S x N x 32
        # Output: S x N x 64, State-H: S x 64, State-C：S x 64
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size]))


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func # "concat"

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError(
                'Unknown attention score function! Must be either dot, general or concat.')
        if attention_func == 'general':
            # General score function
            self.wa = tf.keras.layers.Dense(rnn_size)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = tf.keras.layers.Dense(rnn_size, activation='tanh')
                       # 64 neurons, the input and weights are not decided
            self.va = tf.keras.layers.Dense(1) # 1 neuron, the input and weights are not decided

    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # Dot score function: decoder_output (dot) encoder_output
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            # General score function: decoder_output (dot) (Wa (dot) encoder_output)
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, self.wa(
                encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':
            # Concat score function: va (dot) tanh(Wa (dot) concat(decoder_output + encoder_output))
            # Decoder output must be broadcasted to encoder output's shape first

            temp = encoder_output.shape[1] # 10， the sequence length of encoder input

            # Replicate (S x 1 x 64) to (S x 10 x 64),  10 is the sequence length of encoder input
            # So we can calculate all the alignment scores with 1 operation.
            decoder_output = tf.tile( decoder_output, [1, encoder_output.shape[1], 1] )
                      # Input: S x 1 x 64
                      # Output: S x 10 x 64

            temp1 = tf.concat((decoder_output, encoder_output), axis=-1) # concatenation
                      # Input: S x 10 x 64, S x 10 x 64
                      # Output: S x 10 x 128
            temp2 = self.wa (temp1) # tanh(w x + b),  weights: 64 x 128 + 64
                      # Input: S x 10 x 128
                      # Output: S x 10 x 64
            temp3 = self.va (temp2) # w x + b, weights: 64 + 1
                      # Input: S x 10 x 64
                      # Output: S x 10 x 1

            score = self.va( self.wa( tf.concat((decoder_output, encoder_output), axis=-1)) )
                    # Input: S x 10 x 64, S x 10 x 64
                    # Output: S x 10 x 1

            score = tf.transpose(score, [0, 2, 1])
                    # Input: S x 10 x 1
                    # Output: S x 1 x 10, 10 is the sequence length of encoder input

        # alignment a_t = softmax(score)
        alignment = tf.nn.softmax(score, axis=2) # S x 1 x 10 ( Probabilities)

        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output) #  Sx1x10  x Sx10x64 = S x 1 x 64

        return context, alignment

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size, attention_func):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func) # the attention layer
        self.rnn_size = rnn_size # 64
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size) # 110, 32
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
                                                                     # this is the LSTM layer
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh') # this is the fully-connected layer
        self.ws = tf.keras.layers.Dense(vocab_size)                  # Classifier

    def call(self, sequence, state, encoder_output):
        # This function is used for both training and inference;
        # so for each sample, it can only handle 1 word/decoder-input at one time.
        # For the training, we can introduce the batch processing:
        # - calculate all the LSTM-out for the decoder intput sequences
        # - calculate all the Attention-out
        embed = self.embedding(sequence) # Input：S x 1, ID (0~109)
                                         # Output: S x 1 x 32, Word Vedtor

        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state) # Input: S x 1 x 32
                                       # Output: S x 1 x 64 (lstm_out == state_h)
                                       # State-H: S x 64
                                       # State-C: S x 64

        # Compute the context vector and alignment scores
        context, alignment = self.attention(lstm_out, encoder_output)
        # Input:  S x 1 x 64, S x 10 x 64  (10 is the sequence length of encoder input)
        # Output: S x 1 x 64, 1 x 1 x 10

        temp1 = tf.squeeze(context, 1) # S x 1 x 64 -> S x 64
        temp2 =  tf.squeeze(lstm_out, 1) # S x 1 x 64 -> S x 64

        # Concatenate the intermediate LSTM output and the context
        lstm_out = tf.concat( [tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
                                      # Input: S x 1 x 64 , S x 1 x 64
                                      # Output: S x 128

        # the fully-connected layer: tanh(w x + b),  weights: 64 x 128 + 64
        lstm_out = self.wc(lstm_out)  # Input: S x 128
                                      # Output: S x 64, the final decoder output - representing a word

        # the classifier to convert the decoder output to a word: wx + b, weights: 64 x 110 + 110
        logits = self.ws(lstm_out)   # Input: S x 64
                                     # S x 110

        return logits, state_h, state_c, alignment

#######################################################
# Initialize the model (number of weights and initial values) by feeding in some dummy data (EagerTensor).

en_vocab_size = len(en_tokenizer.word_index) + 1
fr_vocab_size = len(fr_tokenizer.word_index) + 1

EMBEDDING_SIZE = 32
LSTM_SIZE = 64

# Set the score function to compute alignment vectors
# Can choose between 'dot', 'general' or 'concat'
ATTENTION_FUNC = 'concat'

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE, ATTENTION_FUNC)

initial_states = encoder.init_states(1)  # the initial State-H (1x64) and State-C (1x64 ) for the encoder
# Initialize the model with the dummy data: 1 x 10 (2-D，a batch with 1 sample)
encoder_outputs = encoder(tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]), initial_states)
# Initialize the model with the dummy data: 1 x 1
# The Luong-Attention decoder can only handle 1 word / decoder - input at one time.
decoder_outputs = decoder(tf.constant([[1]]), encoder_outputs[1:],encoder_outputs[0])

def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # from_logits=True, the softmax will be applied
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    # mask those zeros out when computing the loss and updating weights (for the decorder output)

    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam()

#@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        temp1 = target_seq_out.shape[1] # 14 is the sequence length of decoder output (or input)
        for i in range(target_seq_out.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            # For each sample, we only handle 1 word/decoder-input at one time.
            temp2 = target_seq_in[:, i] # Get No i word from all S samples,
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1) # S -> S x 1
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in, (de_state_h, de_state_c), en_outputs[0])

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]


def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(de_input, (de_state_h, de_state_c), en_outputs[0])

        temp1 = tf.argmax(de_output, -1) # 1x 110 probabilities -> 1 ID

        de_input = tf.expand_dims( tf.argmax(de_output, -1), 0 ) # 1 -> 1 x 1, the next input will be the current output

        temp2 = de_input.numpy()
        temp3 = de_input.numpy()[0][0]

        out_words.append( fr_tokenizer.index_word[ de_input.numpy()[0][0] ] )

        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return np.array(alignments), test_source_text.split(' '), out_words


NUM_EPOCHS = 30
for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    # Get 5 initial encoder State-H and State-C,  ( 5 x 64, 5 x 64 )
    # Samples are independent to each other, and all samples are handled with the same Encoder LSTM initial states

    # predict("How are you today ?")

    # the dataset includes 4 batches; take(-1) will get all the 4 batches
    #temp1 = dataset.take(-1)
    #temp2 = enumerate(temp1)

    # batch: 0,1,2,3
    # each batch contains 5 samples
    # for each sample:    Encoder Input     Decorder Input       Decorder Output
    #                    (10 English words, 14 French words) -> (14 French words)
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in, target_seq_out, en_initial_states)
        # 5 samples and 5 initial encoder State-H and State-C
        # 5 samples(English) are irrelevant to each other
        # Get the loss from 1 batch with 5 samples
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