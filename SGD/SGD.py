import tensorflow as tf
from tensorflow.keras import layers,optimizers,models
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import plot_model

def ann_model():
    # Define the deep artificial neural network (ANN) model
    ann_model = models.Sequential([
        layers.Flatten(input_shape=(250, 250, 3)),
        layers.Dense(256, activation='relu'),  # Increased nodes
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),  # Added another dense layer
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])

    # Compile the model with Adam optimizer and early stopping
    adam_optimizer = optimizers.SGD(learning_rate=0.001)  # Adjust learning rate
    ann_model.compile(optimizer=adam_optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Print the model summary
    ann_model.summary()
    return ann_model


import os

# Get the absolute path to the directory containing the code file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the relative paths to the data folders
training_data_path = os.path.join(current_directory, 'Preprocess', 'training_data')
validation_data_path = os.path.join(current_directory, 'Preprocess', 'validation_data')
testing_data_path = os.path.join(current_directory, 'Preprocess', 'testing_data')

# Load the data using absolute or relative paths
training_data = tf.data.experimental.load(training_data_path)
validation_data = tf.data.experimental.load(validation_data_path)
testing_data = tf.data.experimental.load(testing_data_path)



model=ann_model()
checkpoint = ModelCheckpoint("ann_sgd_model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history =model.fit(training_data, validation_data=validation_data, epochs = 20, callbacks=callbacks_list)

model_json = model.to_json()
with open("ann_sgd_model.json", "w") as json_file:
    json_file.write(model_json)
import pickle

with open('ann_sgd_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)


