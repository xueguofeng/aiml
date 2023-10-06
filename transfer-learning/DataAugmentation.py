import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
) # Training Dataset:2936, Validation Dataset: 367, Test Dataset: 367

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str



image, label = next( iter(train_ds) )  # get 1 image with its label
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

'''
myIter = iter(train_ds)

image, label = next( myIter )
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

image, label = next( myIter )
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
plt.show()

'''

IMG_SIZE = 180

resize_and_rescale = tf.keras.Sequential([  # define the preprocessing layer
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])


result = resize_and_rescale( image ) # Tensor Input, 333x500 x 3 -> Tensor Output, 180x180 x 3
_ = plt.imshow(result)               # image, EagerTensor        -> result, EagerTensor
plt.show()

print("Min and max pixel values:", result.numpy().min(), result.numpy().max())


data_augmentation = tf.keras.Sequential([   # define the preprocessing layer
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# Add the image to a batch (the batch size is 1). # this line first change(add) a dimension to the tensor for batch, then convert the values of the tensor to float32
image = tf.cast( tf.expand_dims(image, 0), tf.float32 ) # image, 1 x 333x500 x 3

plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image) # Tensor Input(batch) -> Tensor Output(batch)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0]) # the first element of batch [0]
  plt.axis("off")
plt.show()


