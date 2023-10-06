import tensorflow as tf

inputs = tf.keras.Input(shape=(None, None, 3))

processed = tf.keras.layers.RandomCrop(width=32, height=32)(inputs)

conv = tf.keras.layers.Conv2D(filters=2, kernel_size=3)(processed)

pooling = tf.keras.layers.GlobalAveragePooling2D()(conv)

feature = tf.keras.layers.Dense(10)(pooling)

# the above are all kerastensors and are used to construct the model
# none of them are eagertensors


full_model = tf.keras.Model(inputs, feature)
full_model.summary()

backbone = tf.keras.Model(processed, conv)
backbone.summary()

activations = tf.keras.Model(conv, feature)
activations.summary()