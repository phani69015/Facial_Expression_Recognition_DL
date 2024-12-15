# Enhanced Application of Deep Learning Models on Facial Expression Classification

## Project Structure

The folder structure should be organized as follows:


## Steps to Run the Project

### Step 1: Preprocessing
- Run the `preprocess.py` file located in the `Preprocess/` folder to make sure the `training`, `testing`, and `validation` folders are created.

### Step 2: Running Models
- Go to each model folder (e.g., `ANN`, `SEQ_MODEL`, etc.) and run the respective model script (e.g., `ANN.py`, `seq_model.py`, etc.) one by one.
- After running the models, ensure the following three files are saved inside each model folder:
  - `model.json`
  - `model_weights.keras` (Note: Use `.keras` instead of `.h5` for compatibility with the latest versions of PyCharm)
  - `model_history.pkl`

### Step 3: Create Test Models Folder
- Create a subfolder inside the `DL_PROJECT` folder and name it `TEST_MODELS`.
- Include the following Python files in this folder:
  - `test.py`
  - `metrics.py`

### Step 4: Running Test and Metrics Files
- Run the `test.py` and `metrics.py` files.
- The output will show the following:
  1. Seq model
  2. Optimizers
  3. Comparison of optimizers
  4. ANN multiclass
  5. Seq model with mini batch
  6. CNN
  7. CNN with regularization
  8. VGG
  9. RNN
  10. CNN + LSTM
  11. ResNet 50
  
  - Select the model to get predictions of the testing data and training history.

### Step 5: Model Selection and Results
- Select the required model from the interface to view its prediction results and training metrics.
- Exit the interface once you are done with the selection.

### Step 6: Web Application Integration
- Inside the `templates` folder, include the following HTML files:
  - `index.html`
  - `results.html`
  - `testing.html`

- Inside the project folder, create the `app.py` file to integrate the entire project into a web application.

### Step 7: Running the Web Application
- Run `app.py` to start a live server with the integrated project on a web page.

---

## Key Instructions

- **File Names**: The file names are case-sensitive as the operating system used loads files without considering locations.
  
- **Keras Compatibility**: `.h5` is deprecated in the latest versions of PyCharm, so ensure you use `.keras` for saving model weights.

---

### Demonstration

[![Watch the video](https://github.com/phani69015/Facial_Expression_Recognition_DL/blob/master/metrics.png)](https://www.loom.com/share/05a20af0b3d04dbaab5893398cf3f8c9?sid=7043a194-fb28-483f-b92d-7204438b369b)
