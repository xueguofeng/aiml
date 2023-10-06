
from tensorflow import keras



model = keras.models.load_model('c5.mymodel.h5')
config = model.to_json()
print(config)
print(model.summary())