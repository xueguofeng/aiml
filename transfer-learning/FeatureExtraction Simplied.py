import datetime
starttime = datetime.datetime.now()


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

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


val_batches = tf.data.experimental.cardinality(validation_dataset) # 32 batches
test_dataset = validation_dataset.take(val_batches // 5) # 6 batches for test
validation_dataset = validation_dataset.skip(val_batches // 5) # 26 batches for validation


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


#########################################################
# Build the new model

# create a preprocessing layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

# as a preprocessing layer
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# the function from the imported model, from [0, 255] to [-1, 1]

# rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Import the pre-trained model without its classification layer
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
print(base_model.summary())

# Create a new layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# create a new layer without activation; only W X
prediction_layer = tf.keras.layers.Dense(1)


# build the new model based on the defined preprocessing layers and imported base_model
# Use the Keras Functional API
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

#########################################################
# Compile
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print( model.summary() )
print( len(model.trainable_variables) )

#########################################################
# Training
initial_epochs = 2
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


#########################################################
# Time
endtime = datetime.datetime.now()
print(("The running time:")+ str((endtime-starttime).seconds)+" s")


#########################################################
# Show

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()