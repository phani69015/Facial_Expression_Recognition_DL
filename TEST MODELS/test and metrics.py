import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
while True:
    print('1: Seq moedel\n2:Optimizers\n3:Compare with optimizers\n4:ANN_multiclass\n5:seq_model with mini batch\n'
          '6:CNN\n7:CNN_regularization\n8:VGG\n9:RNN\n10:CNN+LSTM\n11:Resnet 50')
    print('Select the model to get predictions of testing data and Training history')
    n = int(input())

    # Dictionary to map model numbers to their respective details
    d = {
        1: ['Seq model', 'Seq_model.json', 'Seq_model_weights.h5', 'seq_training_history.pkl'],
        2: [['ANN_ADAM', 'ann_adam_model.json', 'ann_adam_model_weights.h5', 'ann_adam_training_history.pkl'],
            ['SGD', 'ann_sgd_model.json', 'ann_sgd_model_weights.h5', 'ann_sgd_training_history.pkl'],
            ['RMSPROP', 'rmsprop.json', 'rmsprop_weights.h5', 'rmsprop_history.pkl']],
        4: ['ANN', 'ann_model.json', 'ann_weights.h5','ann_training_history.pkl'],
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

    if n!=3:
        # Handling selection for model type 2
        if n == 2 :
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
        if n!=8:
        # Load JSON architecture of the model
            with open(json_file_path, 'r') as json_file:
                loaded_model_json = json_file.read()

            # Create model using loaded architecture
            model = model_from_json(loaded_model_json)
        # Load pre-trained weights into the model
            model.load_weights(weights)
        else:
            model=tf.keras.models.load_model(weights)

        testing_data_path = os.path.join(project_directory, 'Preprocess', 'testing_data')
        testing_data = tf.data.experimental.load(testing_data_path)

        class_names = [
            'angry',
            'disgust',
            'fear',
            'happy',
            'neutral',
            'sad',
            'surprise'
        ]

        plt.figure(figsize=(30, 30))
        for images, labels in testing_data.take(1):
            predictions = model.predict(images)
            predlabel = []

            for mem in predictions:
                predlabel.append(class_names[np.argmax(mem)])
            plt.subplots_adjust(wspace=0.5, hspace=0.5, top=0.9, bottom=0.1)
            for i in range(40):
                ax = plt.subplot(10, 4, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title('Pred: ' + predlabel[i] + ' actl:' + class_names[labels[i]])
                plt.axis('off')
                plt.grid(True)
            plt.show()

        # Plotting training and validation loss and accuracy
        with open(hist, 'rb') as file:
            loaded_history = pickle.load(file)

        loss = loaded_history['loss']
        accuracy = loaded_history['accuracy']
        val_loss = loaded_history['val_loss']
        val_accuracy = loaded_history['val_accuracy']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        optimizers = ['Adam', 'SGD', 'RMSprop']
        histories = []

        for key in range(2, len(d) + 1):
            if key == 2:
                for item in d[key]:
                    optimizer, json_file, weights_file, history_file = item
                    hist = os.path.join(project_directory, optimizer, history_file)
                    with open(hist, 'rb') as file:
                        history = pickle.load(file)
                        histories.append((optimizer, history))
        # Plotting accuracies and losses
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        for optimizer, history in histories:
            plt.plot(history['accuracy'], label=f'{optimizer} Accuracy')
        plt.title('Training Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        for optimizer, history in histories:
            plt.plot(history['loss'], label=f'{optimizer} Loss')
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()