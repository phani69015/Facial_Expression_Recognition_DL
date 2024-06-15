import tensorflow as tf
from tensorflow.keras import layers,models
from keras.callbacks import ModelCheckpoint
import os


def ann_model():
    # Define the deep artificial neural network (ANN) model
    ann_model = models.Sequential([
        layers.Flatten(input_shape=(250, 250, 3)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])

    # Compile the model without specifying an optimizer
    ann_model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Print the model summary
    ann_model.summary()
    return ann_model




current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specify the relative paths to the data folders
training_data_path = os.path.join(current_directory, 'Preprocess', 'training_data')
validation_data_path = os.path.join(current_directory, 'Preprocess', 'validation_data')
testing_data_path = os.path.join(current_directory, 'Preprocess', 'testing_data')

# Load the data using absolute or relative paths
training_data = tf.data.experimental.load(training_data_path)
validation_data = tf.data.experimental.load(validation_data_path)
testing_data = tf.data.experimental.load(testing_data_path)



model=ann_model()
checkpoint = ModelCheckpoint("ann_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history =model.fit(training_data, validation_data=validation_data, epochs = 20, callbacks=callbacks_list)

model_json = model.to_json()
with open("ann_model.json", "w") as json_file:
    json_file.write(model_json)
import pickle

with open('ann_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)


