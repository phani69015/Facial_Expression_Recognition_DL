import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import os


def create_facial_expression_model(input_shape=(250,250,3)):
    # Define your sequential model
    model = Sequential()

    # Add Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # Add Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

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




model=create_facial_expression_model(input_shape=(250,250,3))
checkpoint = ModelCheckpoint("minibatch_model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history =model.fit(training_data,validation_data=validation_data, batch_size=32, epochs=5, validation_split=0.1, callbacks=callbacks_list)

model_json = model.to_json()
with open("minibatch_model.json", "w") as json_file:
    json_file.write(model_json)
import pickle

with open('minibatch_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)














