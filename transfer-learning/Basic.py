import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# https://www.tensorflow.org/tutorials/images/transfer_learning#continue_training_the_model

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train') # 1000 dogs and 1000 cats
validation_dir = os.path.join(PATH, 'validation') # 500 dogs and 500 cats

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# 2000 dogs and cats， 63 batches (62.5)
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# 1000 dogs and cats, 32 batches (31.25)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
# 0 for 'cats', 1 for 'dogs'
class_names = train_dataset.class_names


plt.figure(figsize=(10, 10))  # 10 inches x 10 inches

for images, labels in train_dataset.take(1):  # get 1 batch, 32 images with 32 labels
                                              # images and lables are the EagerTensor type, the for loop only runs 1 time
  for i in range(BATCH_SIZE):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow( images[i].numpy().astype("uint8") )
    plt.title( class_names[labels[i]] )
    plt.axis("off")
plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset) # 32 batches

temp = val_batches // 5

test_dataset = validation_dataset.take(val_batches // 5) # 6 batches for test
validation_dataset = validation_dataset.skip(val_batches // 5) # 26 batches for validation

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


AUTOTUNE = tf.data.AUTOTUNE
# two tasks for prefetch, current 32 being augmented and sent into forward propogation, the next 32 getting ready already
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


for image, _ in train_dataset.take(1):  # get 1 batch, 32 images with 32 labels
                                        # image is the EagerTensor type, the for loop only runs 1 time
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation (tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
#plt.show()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print(preprocess_input) # the function from the model, we will not use it

rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2, which is pretrained using ImageNet Dataset and consist of 1.4M images and 1000 classes
# we are doing features retraction, not fine-tuning
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
       # We don't need the last classification layer from this pre-trained model.
       # We only need the pre-trained model for feature extraction: from the original image to a vector
       # We will create our own classification layer based on the vector ( extracted feature from the original image).


image_batch, label_batch = next( iter(train_dataset) ) # Get a batch - EagerTesnor 32x 160x160  x 3
feature_batch = base_model(image_batch) # 32 x 160x160x3 -> 32 x 5x5x1280
print(feature_batch.shape)

print("-------------------------------------------")

