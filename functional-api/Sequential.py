import tensorflow as tf


model = tf.keras.Sequential() # type = sequential

# Optionally, the first layer can receive an `input_shape` argument:
model.add(tf.keras.layers.Dense(8, input_shape=(16,)))

# Afterwards, we do automatic shape inference:
model.add(tf.keras.layers.Dense(4))


model.compile(optimizer='sgd', loss='mse')



model.summary()

# check tf.keras.Sequential == tf.keras.models.Sequential in python terminal, result = true