
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import  Flatten, Dense
from tensorflow.keras.models import Model
import os
from keras.callbacks import ModelCheckpoint



def vgg():
    vgg = VGG16(input_shape=[250, 250] + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    vgg16_model = keras.Sequential([
        vgg,
        Flatten(),
        Dense(2, activation='relu'),
        Dense(1, activation='sigmoid')])
    vgg16_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    vgg16_model.summary()
    return vgg16_model

# Create VGG model
custom_model = vgg()

# Set current directory to the "DL project" directory
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specify the relative paths to the data folders
training_data_path = os.path.join(current_directory, 'Preprocess', 'training_data')
validation_data_path = os.path.join(current_directory, 'Preprocess', 'validation_data')
testing_data_path = os.path.join(current_directory, 'Preprocess', 'testing_data')

# Load the data using absolute or relative paths
training_data = tf.data.experimental.load(training_data_path)
validation_data = tf.data.experimental.load(validation_data_path)
testing_data = tf.data.experimental.load(testing_data_path)




checkpoint = ModelCheckpoint("vgg_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 5, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("vgg_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('vgg_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
