# To recognize cats and dogs
# Training data: 1000 cats and 1000 dogs
# Test data: 500 cats and 500 dogs


import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory of Data
base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Training Data
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Test Data
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

print(train_cats_dir)
print(train_dogs_dir)
print(validation_cats_dir)
print(validation_dogs_dir)


model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

temp = model.summary()
print(temp)

model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255) # convert 0~255 to 0~1
test_datagen = ImageDataGenerator(rescale=1./255)  # convert 0~255 to 0~1

# Provide metadata to generate data from the directory
train_generator = train_datagen.flow_from_directory(
        train_dir,  # directory
        target_size=(64, 64),  # convert the real picture size into the target size
        batch_size=20,         # the size of mini batch, 20 pictures for each forward propagation and back propagation
        # Label: one-hot or binary (only 2 classes)
        # 'categorical' for one-hot encoding, 'binary' for 2 classes
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')

# instead of loading all data files into memory, load the necessary data for each iteration dynamically
history = model.fit_generator(
      train_generator,
      steps_per_epoch = 100,  # during 1 epoch, 20 (mini-batch) x 100 (iteration) = 2000, all training data
      epochs = 20, # 20 times to process all training data
      validation_data = validation_generator,
      validation_steps = 50,  # 1000 / batch_size 20
      verbose = 2)


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()