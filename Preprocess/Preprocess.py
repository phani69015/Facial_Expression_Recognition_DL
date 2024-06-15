import tensorflow as tf
import os
import gdown
import zipfile

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



# Google Drive file ID
file_id = '1BAIFYfQuwnrWP8CU9giHJbClQZfVvaEp'

# Destination path where the file will be saved
output_zip = 'facial_ex_dataset.zip'

# Download the file from Google Drive
gdown.download('https://drive.google.com/uc?id=' + file_id, output_zip, quiet=False)

# Destination directory to extract the zip file
output_folder = 'facial_ex_dataset'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Unzip the downloaded file
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(output_folder)

# Remove the downloaded zip file if needed
os.remove(output_zip)

print("Dataset downloaded and extracted successfully.")


# Set current directory to the "DL project" directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the relative paths to the data folders
train_path = os.path.join(current_directory,'facial_ex_dataset', 'facial_ex_dataset', 'train')
val_path = os.path.join(current_directory,'facial_ex_dataset', 'facial_ex_dataset', 'val')
test_path = os.path.join(current_directory, 'facial_ex_dataset','facial_ex_dataset', 'test')

training_data, validation_data, testing_data, class_names = load_and_configure_datasets(train_path, val_path, test_path)

import numpy as np

# Save datasets
tf.data.Dataset.save(training_data, 'training_data')
tf.data.Dataset.save(validation_data, 'validation_data')
tf.data.Dataset.save(testing_data, 'testing_data')

# Save class names
np.save('class_names.npy', class_names)

