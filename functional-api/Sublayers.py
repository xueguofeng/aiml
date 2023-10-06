import tensorflow as tf
from tensorflow.keras import layers

class Linear(tf.keras.layers.Layer): # linear is like Circle, keras layer is like shape,keras layer has __call__ and it calls call()
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs): # redraw
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2) # initialized the layer, calls the constructor in the class and set all the weights and bias with given parameters (4,2)
y = linear_layer(x) # when ready to use the layer, it calls the call function, the x is the real input and this is a legit computational step
# linear_layer() is the python built in __call__
print(y)