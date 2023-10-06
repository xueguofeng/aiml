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


'''
plt.figure(figsize=(10, 10))  # 10 inches x 10 inches


for images, labels in train_dataset.take(1):  # get 1 batch, 32 images with 32 labels
                                              # images and lables are the EagerTensor type, the for loop only runs 1 time
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow( images[i].numpy().astype("uint8") )
    plt.title( class_names[labels[i]] )
    plt.axis("off")
plt.show()
'''

val_batches = tf.data.experimental.cardinality(validation_dataset) # 32 batches

test_dataset = validation_dataset.take(val_batches // 5) # 6 batches for test
validation_dataset = validation_dataset.skip(val_batches // 5) # 26 batches for validation

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


'''
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
'''


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print(preprocess_input) # the function from the model, from [0, 255] to [-1, 1]


rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)  # from [0, 255] to [-1, 1]

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
       # We don't need the last classification layer from this pre-trained model.
       # We only need the pre-trained model for feature extraction: from the original image to a vector
       # We will create our own classification layer based on the vector ( extracted feature from the original image).



# 实际计算，不是做模型
# 取出1个Batch，32个Image及label，输入基础模型，产生32个输出
image_batch, label_batch = next( iter(train_dataset) ) # Get a batch - EagerTesnor 32x 160x160  x 3
feature_batch = base_model(image_batch) # 输入input：32 x 160x160x3 -> 输出output：32 x 5x5x1280
print(feature_batch.shape)              # Do the computation actually, not to build a model


print("-------------------------------------------")


base_model.trainable = False
# the base model is only a layer in our new model, and frozen - not trainable

print(base_model.summary())


# 实际计算，不是做模型
# 输入input：32 x 160x160x3 -> 输出output：32 x 5x5x1280 -> 取平均：32 x 1280
global_average_layer = tf.keras.layers.GlobalAveragePooling2D() # create a new layer
feature_batch_average = global_average_layer(feature_batch)  # 32 x 5x5x1280 -> 32 x 1280
print(feature_batch_average.shape)     # Do the computation actually, not to build a model

# 实际计算，不是做模型
# 采用1个Dense层（无Activation），进行二分类
# 输入input：32 x 160x160x3 -> 输出output：32 x 5x5x1280 -> 取平均：32 x 1280 -> 32个Value
prediction_layer = tf.keras.layers.Dense(1) # create a new laye r, without activation; only W X
prediction_batch = prediction_layer(feature_batch_average) # 32 x 1280 -> 32 values
print(prediction_batch.shape)           # Do the computation actually, not to build a model
# print(prediction_batch)


# build the new model based on the defined preprocessing layers and imported base_model
# Use the Keras Functional API
# Not do the computation here, just to build the model

inputs = tf.keras.Input(shape=(160, 160, 3))  # 构造Input，为KerasTensor
x = data_augmentation(inputs)       # 预处理
x = preprocess_input(x)             # 预处理
x = base_model(x, training=False)   # 基础模型
x = global_average_layer(x)         # 取平均
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)       # 输出层

model = tf.keras.Model(inputs, outputs)  # 建模型

# 进行编译，提供学习率、梯度下降方法及LOSS
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print( model.summary() )
print( len(model.trainable_variables) ) # only 2 trainable variables
                                        # tf.Variable: w (1280), b (1)

initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset) # evaluate the model without training

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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