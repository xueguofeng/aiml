import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# link to where all the pictures are stored
baseDirectory = './data/cats_and_dogs'
trainDirectory = os.path.join(baseDirectory, 'train')
validationDirectory = os.path.join(baseDirectory, 'validation')
catTrainingDirectory = os.path.join(trainDirectory, 'cats')
dogTrainingDirectory = os.path.join(trainDirectory, 'dogs')
catValidationDirectory = os.path.join(validationDirectory, 'cats')
dogValidationDirectory = os.path.join(validationDirectory, 'dogs')

# initialize the CNN model and its hidden layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['acc'])

# image augmentation and rescaling RGB 255 for training set
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# only rescaling RGB 255 for training set
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        trainDirectory,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validationDirectory,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')


filepath = 'c5.mymodel.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=2)
callbacks_list = [checkpoint] # define the callback which contains several checkpoints


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images / 20 batch-size
      epochs=20, # 20 times
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2, callbacks=callbacks_list)   # provide the callback to save the model

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()