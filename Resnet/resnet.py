import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import os
from keras.callbacks import ModelCheckpoint
def resnet_block(inputs, num_filters, kernel_size, strides, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    shortcut = Conv2D(num_filters, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x

def resnet(input_shape=(250, 250, 3), num_classes=7):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = resnet_block(x, num_filters=64, kernel_size=(3, 3), strides=(1, 1))
    x = resnet_block(x, num_filters=64, kernel_size=(3, 3), strides=(1, 1))

    x = resnet_block(x, num_filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = resnet_block(x, num_filters=128, kernel_size=(3, 3), strides=(1, 1))

    x = resnet_block(x, num_filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = resnet_block(x, num_filters=256, kernel_size=(3, 3), strides=(1, 1))

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Create ResNet model
custom_model = resnet(input_shape=(250, 250, 3))
custom_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

custom_model.summary()



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




checkpoint = ModelCheckpoint("resnet_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 2, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("resnet_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('resnet_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)