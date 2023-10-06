import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

vocab = ["a", "b", "c", "d"]
data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])

layer = layers.StringLookup(vocabulary=vocab)  # the internal state for this preprocessing layer

vectorized_data = layer(data) # call the layer
print(vectorized_data)