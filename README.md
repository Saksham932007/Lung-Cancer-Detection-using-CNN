# 🫁 Lung Cancer Detection using CNN

This project showcases a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify histopathological images of lung tissue into three categories: **Normal**, **Adenocarcinoma**, and **Squamous Cell Carcinoma**. This is a powerful application of computer vision in the medical field to aid in the early detection of lung cancer.

[](https://www.python.org/downloads/)
[](https://www.tensorflow.org/)
[](https://keras.io/)
[](https://opensource.org/licenses/MIT)

[](https://www.google.com/search?q=https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/Untitled13.ipynb)

> **Note:** Replace `YOUR_USERNAME/YOUR_REPO_NAME` in the URL above with your actual GitHub username and repository name to make the "Open in Colab" badge work.

-----

## Project Overview

The goal of this project is to develop a deep learning model capable of accurately classifying lung tissue images. By leveraging a CNN, the model learns to identify complex patterns and features from the images that distinguish between healthy and cancerous cells.

### Workflow

The project follows a standard machine learning pipeline:

-----

## Dataset

The model is trained on the **"Lung and Colon Cancer Histopathological Images"** dataset available on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images). It contains 5,000 images for each of the three lung tissue categories.

Here are some sample images from the dataset:

-----

## Technology Stack

  * **Programming Language:** Python
  * **Libraries:**
      * TensorFlow & Keras (for building and training the CNN)
      * Scikit-learn (for data splitting and evaluation metrics)
      * OpenCV (for image processing)
      * Pandas & NumPy (for data manipulation)
      * Matplotlib (for data visualization)

-----

## Model Architecture

A sequential CNN model was designed with the following layers:

  * Three `Conv2D` layers with `ReLU` activation, each followed by a `MaxPooling2D` layer to extract features.
  * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
  * Two `Dense` (fully connected) layers with `BatchNormalization` and `Dropout` to prevent overfitting.
  * A final `Dense` output layer with `Softmax` activation for multi-class classification.

Here is a summary of the model:

-----

## Results and Evaluation

The model was trained for 10 epochs. The training process included callbacks like `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to adjust the learning rate.

### Training Performance

The plot below shows the training and validation accuracy over the epochs. The model learns the training data well, while the validation accuracy shows how it generalizes to unseen data.

### Classification Report

The model's performance on the validation set is summarized in the classification report below. It achieved an overall accuracy of **72%**.

**Observations:**

  * The model performs exceptionally well in identifying normal lung tissue (`lung_n`) with a precision of 0.99 and a recall of 0.84.
  * It perfectly identifies all squamous cell carcinoma cases (`lung_scc`), achieving a recall of 1.00, though with lower precision (0.58).
  * The model struggles the most with adenocarcinoma (`lung_aca`), indicating that this class is harder to distinguish.

-----

## How to Run This Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  **Set up Kaggle API:**

      * Download your `kaggle.json` API token from your Kaggle account settings.
      * When you run the notebook, it will prompt you to upload this file.

3.  **Run the Notebook:**

      * Open `Untitled13.ipynb` in Google Colab or a local Jupyter environment.
      * Run the cells sequentially to download the data, train the model, and evaluate its performance.

-----

## Future Improvements

  * **Data Augmentation:** Apply techniques like rotation, flipping, and zooming to artificially increase the dataset size and improve model generalization.
  * **Transfer Learning:** Use a pre-trained model (like ResNet, VGG16, or InceptionV3) and fine-tune it on this dataset to potentially achieve higher accuracy.
  * **Hyperparameter Tuning:** Systematically tune parameters like learning rate, batch size, and the number of layers to optimize performance.

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
