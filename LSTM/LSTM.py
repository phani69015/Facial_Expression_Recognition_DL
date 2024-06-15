import tensorflow as tf
from tensorflow.keras import layers,optimizers,models,Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import plot_model

def lstm(learning_rate=0.001):
    model = models.Sequential([
        layers.Reshape((1, -1), input_shape=(250, 250, 3)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(7, activation='softmax')  # 5 classes for your case
    ])

    # Compile the model with Adam optimizer
    adam = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return model


custom_model = lstm()


import os

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





checkpoint = ModelCheckpoint("lstm_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 10, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("lstm_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('lstm_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)