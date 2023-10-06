import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))

x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)  # first call constructor and initialize, then plug in inputs and call the call function

outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs) # inputs can be x, but that will result in 2 layers

model.summary()