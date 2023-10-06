

import collections
import os
import random
import urllib
import zipfile

import numpy as np
import tensorflow as tf


# training parameters
learning_rate = 0.1
batch_size = 128 # 128 word pairs each time, 64 x 2
num_steps = 300000 # epochs
display_step = 10000
eval_step = 200000

# test data
eval_words = ['nine', 'of', 'going', 'hardware', 'american', 'britain']

# Word2Vec parameters
embedding_size = 200 # word vector dimension
max_vocabulary_size = 50000
min_occurrence = 10 # least word frequency
skip_window = 3 #
num_skips = 2 # randomly select 2 from the window
num_sampled = 64 # negative sampling #

# the original corpus
data_path = 'text8.zip'
with zipfile.ZipFile(data_path) as f:
    text_words = f.read(f.namelist()[0]).lower().split()

temp = len(text_words) # contain 17,005,207 words in total (and 253854 unique words)
print(temp)

# A counter
count = [('UNK', -1)]
# the 49999 high-frequency words
count.extend( collections.Counter(text_words).most_common(max_vocabulary_size - 1) )
# until now, count[0] - ('UNK', -1) is not right
# all the words which are not selected, we need to calculate the total num

temp  = count[49990:50000]
print(temp)

# only keep the words which min_occurrence >= 10
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    else:
        break

vocabulary_size = len(count)  # finally select 47135 words, 47134 + 1 (UNK)

# from word to id (index)
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

# from id(index) to word
data = list()
unk_count = 0
for word in text_words:   #  traverse the corpus, get the ID of each word and save it into the 'data'
    index = word2id.get(word, 0) # Get the ID of each word
                                 # Return 0 if the key(word) doesn't exist in the dict
    if index == 0:
        unk_count += 1
    data.append(index)
count[0] = ('UNK', unk_count) # totally 444176 words in the corpus are not in the vocabulary and replaced with "UNK"
id2word = dict( zip(word2id.values(), word2id.keys()) )

print("Words count:", len(text_words))
print("Unique words:", len(set(text_words)))
print("Vocabulary size:", vocabulary_size)
print("Most common words:", count[:10])


data_index = 0

def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # get window size (words left and right + current one).
    span = 2 * skip_window + 1 # 7
    buffer = collections.deque(maxlen=span)

    if data_index + span > len(data): # reach to the end of corpus, go to the beginning
        data_index = 0

    buffer.extend( data[data_index:data_index + span] )
    data_index += span

    temp = batch_size // num_skips
    for i in range(temp):
        # if batch_size is 128 and only generate 2 samples in a window, we need to slide the window 64 times
    #for i in range(batch_size // num_skips):

        context_words = [w for w in range(span) if w != skip_window] # [0, 1, 2, 4, 5, 6], 3 is input
        words_to_use = random.sample(context_words, num_skips) # select 2 words as output
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window] # 3 is input word
            labels[i * num_skips + j, 0] = buffer[context_word] # output word

        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index]) # slide the window by 1
            data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random.normal([vocabulary_size, embedding_size])) #维度：47135, 200
    nce_weights = tf.Variable(tf.random.normal([vocabulary_size, embedding_size]))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


def get_embedding(x):
    with tf.device('/cpu:0'):
        x_embed = tf.nn.embedding_lookup(embedding, x)
        return x_embed

def nce_loss(x_embed, y):
    with tf.device('/cpu:0'):
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss( weights=nce_weights,biases=nce_biases,labels=y,inputs=x_embed,
                            num_sampled=num_sampled, # how many negative samples generated
                            num_classes=vocabulary_size)
                           )
        return loss


# Evaluation.
def evaluate(x_embed):
    with tf.device('/cpu:0'):
        # Compute the cosine similarity between input data embedding and every embedding vectors
        x_embed = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))#归一化
        embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)#全部向量的
        cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)#计算余弦相似度
        return cosine_sim_op

# SGD
optimizer = tf.optimizers.SGD(learning_rate)


# 迭代优化
def run_optimization(x, y):
    with tf.device('/cpu:0'):

        with tf.GradientTape() as g:
            emb = get_embedding(x)   # get the word embedding
            loss = nce_loss(emb, y)  # Calculate the loss

        # Get the gradients
        gradients = g.gradient(loss, [embedding, nce_weights, nce_biases])

        # Update the gradents
        optimizer.apply_gradients( zip(gradients, [embedding, nce_weights, nce_biases]) )



# test data
x_test = np.array([word2id[w.encode('utf-8')] for w in eval_words])


# training
for step in range(1, num_steps + 1): #

    batch_x, batch_y = next_batch(batch_size, num_skips, skip_window)
    # slide the windows 64 times and generate 128 samples - (input word, output word)

    run_optimization(batch_x, batch_y)  # forward propagation and backward propagation

    if step % display_step == 0 or step == 1:
        loss = nce_loss(get_embedding(batch_x), batch_y)
        print("step: %i, loss: %f" % (step, loss))

    # Evaluation.
    if step % eval_step == 0 or step == 1:
        print("Evaluation...")
        sim = evaluate(get_embedding(x_test)).numpy()
        for i in range(len(eval_words)):
            top_k = 8  # 8 most similar words
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words[i]
            for k in range(top_k):
                log_str = '%s %s,' % (log_str, id2word[nearest[k]])
            print(log_str)