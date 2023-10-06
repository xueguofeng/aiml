import tensorflow as tf
from tensorflow.keras import layers

layer = layers.Dense(32, activation='relu') # create a layer

inputs = tf.random.uniform( shape=(10, 20) )

outputs = layer(inputs)  # A layers is callable, just like a function

print(inputs)
print(outputs)