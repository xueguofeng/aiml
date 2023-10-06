import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a data augmentation stage with horizontal flipping, rotations, zooms
# These input processing pipelines can be used as independent preprocessing code in non-Keras workflow
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
classes = 10

# Create a tf.data pipeline of augmented images (and their labels)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))


# Create a model and train it on the augmented image data
inputs = keras.Input(shape=input_shape)

x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale inputs

outputs = keras.applications.ResNet50(  # Add the rest of the model
    weights=None, input_shape=input_shape, classes=classes)(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

model.fit(train_dataset, steps_per_epoch=20)