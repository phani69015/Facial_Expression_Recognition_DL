import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
from keras.callbacks import ModelCheckpoint

def rnn_model(learning_rate=0.001):
    # Define the Recurrent Neural Network (RNN) model with LSTM units
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(250, 250, 3)),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Reshape((1, -1)),
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


custom_model = rnn_model()

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





checkpoint = ModelCheckpoint("rnn_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 10, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("rnn_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('rnn_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)