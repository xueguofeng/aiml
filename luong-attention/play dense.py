import tensorflow as tf
from tensorflow.keras import layers

# Create a fully-connected layer with 4 neurons
# The weights are not decided because the input is not provided (EagerTensor or KerasTensor)
layer = layers.Dense(4, activation='relu')

# 1 batch with 2 samples
# Each samples has 8 values
inputs = tf.random.uniform( shape=(2, 8) )
print(inputs)

# A layers is callable, just like a function
# The input is either EagerTensor or KerasTensor
outputs = layer(inputs)   # the weights are decided now: ( 8 + 1 ) x 4 = 36
print(outputs)            # Has nothing to do with the batch-size, 2
                          # The output is 2 x 4

# If the input is EagerTensor, it is for computing and the output is EagerTensor
# If the input is KerasTensor, it is for modeling and the output is KerasTensor

print()
# 1 batch with 2 samples, and each samples has 6 values
inputs = tf.random.uniform( shape=(2, 6) )
print(inputs)

#outputs = layer(inputs) # Error, because the number of weight are already fixed
# print(outputs)


print()
# 1 batch with 4 samples, and each samples has 8 values
inputs = tf.random.uniform( shape=(4, 8) )
print(inputs)
outputs = layer(inputs)
print(outputs)         # The output is 4 x 4, [[]] -> [[]]

print()
# 1 batch with 2x2 samples, and each samples has 8 values
inputs = tf.random.uniform( shape=(2, 2 ,8) )
print(inputs)
outputs = layer(inputs)
print(outputs)         # The output is 2 x 2 x 4,  [[[]]] -> [[[]]]

print()
# 1 batch with 1 samples, and each samples has 8 values
inputs = tf.random.uniform( shape=(1, 1 ,8) )
print(inputs)
outputs = layer(inputs)
print(outputs)         # The output is 1 x 1 x 4,  [[[]]] -> [[[]]]

