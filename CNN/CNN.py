import tensorflow as tf
from tensorflow.keras import layers,optimizers,models
from keras.callbacks import ModelCheckpoint

def load_and_configure_datasets(train_path, val_path, test_path, img_height=250, img_width=250, batch_size=100):
    # Loading training set
    training_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='rgb'
    )

    # Extracting class names before prefetching
    class_names = training_data.class_names

    # Configuring dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    training_data = training_data.cache().prefetch(buffer_size=AUTOTUNE)

    # Loading validation dataset
    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='rgb'
    )

    # Loading testing dataset
    testing_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='rgb'
    )

    # Configuring dataset for performance
    validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
    testing_data = testing_data.cache().prefetch(buffer_size=AUTOTUNE)

    return training_data, validation_data, testing_data, class_names




# Example usage:
train_path = 'C:\\Users\\TIRUMALA PHANENDRA\\PycharmProjects\\Deeplearning project\\facial_ex_dataset\\train'
val_path = 'C:\\Users\\TIRUMALA PHANENDRA\\PycharmProjects\\Deeplearning project\\facial_ex_dataset\\val'
test_path = 'C:\\Users\\TIRUMALA PHANENDRA\\PycharmProjects\\Deeplearning project\\facial_ex_dataset\\test'

training_data, validation_data, testing_data, class_names = load_and_configure_datasets(train_path, val_path, test_path)





def cnn_model():
    # Specify input shape and number of classes
    input_shape = (250, 250, 3)
    num_classes = 7

    # Define the convolutional neural network (CNN) model
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
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



checkpoint = ModelCheckpoint("cnn_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = custom_model.fit(training_data, validation_data=validation_data, epochs = 10, callbacks=callbacks_list)


model_json = custom_model.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_json)

import pickle
with open('cnn_training_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)