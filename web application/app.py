import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import json

app = Flask(__name__)

def plot_metrics(loss, accuracy, val_loss, val_accuracy):
    # Plotting training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Create a directory if it doesn't exist
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # Save the plot as a temporary file
    plot_path = os.path.join(static_folder, 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    n = int(request.form['model'])

    d = {
        1: ['Seq model', 'Seq_model.json', 'Seq_model_weights.h5', 'seq_training_history.pkl'],
        2: [['ANN_ADAM', 'ann_adam_model.json', 'ann_adam_model_weights.h5', 'ann_adam_training_history.pkl'],
            ['SGD', 'ann_sgd_model.json', 'ann_sgd_model_weights.h5', 'ann_sgd_training_history.pkl'],
            ['RMSPROP', 'rmsprop.json', 'rmsprop_weights.h5', 'rmsprop_history.pkl']],
        4: ['ANN', 'ann_model.json', 'ann_weights.h5', 'ann_training_history.pkl'],
        5: ['Mini batch', 'minibatch_model.json', 'minibatch_model_weights.h5', 'minibatch_training_history.pkl'],
        6: ['CNN', 'cnn_model.json', 'cnn_weights.h5', 'cnn_training_history.pkl'],
        7: ['CNN_REG', 'cnn_reg_model.json', 'cnn_reg_weights.h5', 'cnn_reg_training_history.pkl'],
        8: ['VGG', 'vgg_model.json', 'vgg_model.h5', 'vgg_training_history.pkl'],
        9: ['RNN', 'rnn_model.json', 'rnn_weights.h5', 'rnn_training_history.pkl'],
        10: ['LSTM', 'lstm_model.json', 'lstm_weights.h5', 'lstm_training_history.pkl'],
        11: ['Resnet', 'resnet_model.json', 'resnet_weights.h5', 'resnet_training_history.pkl'],
    }

    current_directory = os.path.dirname(os.path.abspath(__file__))
    project_directory = os.path.dirname(current_directory)

    if n != 3:
        # Handling selection for model type 2
        if n == 2:
            print('Choose the model\n')
            print('1: ANN_Adam\n', '2: Ann_sgd\n', '3: ANN_rmsprop\n')
            x = int(input())
            if x in range(1, 4):
                json_file_path = os.path.join(project_directory, d[2][x - 1][0], d[2][x - 1][1])
                weights = os.path.join(project_directory, d[2][x - 1][0], d[2][x - 1][2])
                hist = os.path.join(project_directory, d[2][x - 1][0], d[2][x - 1][3])
            else:
                print('INVALID')
        else:
            json_file_path = os.path.join(project_directory, d[n][0], d[n][1])
            weights = os.path.join(project_directory, d[n][0], d[n][2])
            hist = os.path.join(project_directory, d[n][0], d[n][3])
        if n != 8:
            # Load JSON architecture of the model
            with open(json_file_path, 'r') as json_file:
                loaded_model_json = json_file.read()

            # Create model using loaded architecture
            model = model_from_json(loaded_model_json)
            # Load pre-trained weights into the model
            model.load_weights(weights)
        else:
            model = tf.keras.models.load_model(weights)
            # Plotting training and validation loss and accuracy
    with open(hist, 'rb') as file:
        loaded_history = pickle.load(file)
    # Access metrics from loaded history
    loss = loaded_history['loss']
    accuracy = loaded_history['accuracy']
    val_loss = loaded_history['val_loss']
    val_accuracy = loaded_history['val_accuracy']

    # Convert metrics data into JSON format
    loss_json = json.dumps(loss)
    accuracy_json = json.dumps(accuracy)
    val_loss_json = json.dumps(val_loss)
    val_accuracy_json = json.dumps(val_accuracy)

    # Plot metrics
    plot_path = plot_metrics(loss, accuracy, val_loss, val_accuracy)

    # Render the results template with appropriate data
    return render_template('results.html', loss=loss_json, accuracy=accuracy_json, val_loss=val_loss_json, val_accuracy=val_accuracy_json, plot_path=plot_path)

@app.route('/testing', methods=['POST'])
def testing():
    n = int(request.form['model'])
    testing_output = ""

    # Add testing code based on selected model
    if n in [1, 2, 4, 5, 6, 7, 8]:
        # Testing code for classification models
        testing_output = "Add your classification testing output here."
    elif n in [3, 9]:
        # Testing code for autoencoder models
        testing_output = "Add your autoencoder testing output here."

    return render_template('testing.html', testing_output=testing_output)

if __name__ == '__main__':
    app.run(debug=True)
