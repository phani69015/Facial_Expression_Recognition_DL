import tensorflow as tf
from tensorflow.keras import layers,optimizers,models
from keras.callbacks import ModelCheckpoint
import os







def cnn_model():
    # Specify input shape and number of classes
    input_shape = (250, 250, 3)
    num_classes = 7

    # Define the convolutional neural network (CNN) model
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),  # Dropout layer for regularization
        tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),  # Dropout layer for regularization
        tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),  # Dropout layer for regularization
        tf.keras.layers.Conv2D(256, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),  # Dropout layer for regularization
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),   # Dropout layer for regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Build the model with the specified input shape
    cnn_model.build((None,) + input_shape)

    # Print the model summary
    cnn_model.summary()

    return cnn_model

custom_model=cnn_model()


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




checkpoint = ModelCheckpoint("cnn_reg_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 5, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("cnn_reg_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('cnn_reg_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)