ENHANCED APPLICATION OF DEEP LEARNING MODELS ON FACIAL EXPRESSION CLASSIFICATION
---------------------------------------------------------------------------------------------------------------------
User Manual : The basic folders and subfolders format should be as mentioned below

DL PROJECT:
    Facial_ex_dataset
    Preprocess - Preprocess.py,visualize.py
    // Creating folders for different models and including pyhton files inside them
    1. ANN - ANN.py
    2. ANN_ADAM - ANN_ADAM.py
    3. ANN_SGD - ANN_SGD.py
    4. ANN_RMSPROP - ANN_RMSPROP.py
    5. SEQ_MODEL - seq_model.py
    6. MINI_Batch - mini_batch.py
    7. CNN - CNN.py
    8. CNN_REG - cnn_reg.py
    9. VGG- vgg.py
    10. RNN - rnn.py
    12. LSTM - lstm.py
    13. Resnet - resnet.py

step 1      :  run the preprocess python file and make sure training,testing and validation folders are created
step 2      :  Go to each model folder and run the models one by one making sure 3 additionals files are being
               saved inside the model folder
               namely : model.json,model_weights.h5,model_history.pkl
step 3      :  Create a subfolder inside DL project folder and name it as TEST MODELS include a test and metrics
               python file inside it
step 4      :  Run the test and metrics file , the output will be as shown below:
                1: Seq moedel
                2:Optimizers
                3:Compare with optimizers
                4:ANN_multiclass
                5:seq_model with mini batch
                6:CNN
                7:CNN_regularization
                8:VGG
                9:RNN
                10:CNN+LSTM
                11:Resnet 50
                Select the model to get predictions of testing data and Training history

step 5      : Select the required model to get its prediction results and training metrics as long as you exit from
              interface



step 6      : web application integration
              - templates
                 - index.html
                 - results.html
                 - testing.html
              - app.py

step 7      : Run the app.py to get a live server containing the integration of the project in web page.


// key instructions

--> file names are case sensitive as OS is used to load everything without considering locations
--> h5 is depreciated for latest pycharm edition so kindly change it to keras

------------------------------------------------------------------------------------------------------------------------























