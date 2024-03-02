# KNN Classifier for Handwritten Digit Recognition

## Summary:
This code implements a k-nearest neighbors (KNN) classifier for recognizing handwritten digits using the MNIST dataset. It employs the scikit-learn library for data handling, model training, hyperparameter tuning, and performance evaluation.

## Description:
The code consists of the following key components:

1. **Data Loading and Preprocessing**: The MNIST dataset containing grayscale images of handwritten digits and their corresponding labels is loaded using scikit-learn's `load_digits()` function. The dataset is then preprocessed to prepare the feature matrix `X` and target vector `y`.

2. **Data Visualization**: The first ten images from the dataset are visualized to provide an overview of the handwritten digits using Matplotlib.

3. **Model Training and Hyperparameter Tuning**: The dataset is split into training and testing sets using the `train_test_split` function. Hyperparameters for the KNN classifier, such as the number of neighbors (`n_neighbors`), the weight function (`weights`), and the distance metric (`p`), are optimized using `GridSearchCV` to maximize classification accuracy.

4. **Model Evaluation**: The best hyperparameters obtained from the grid search are used to train the KNN classifier (`knn_best`). The accuracy of the classifier is evaluated on the testing set, and a graph showing the accuracy for different values of `k` (number of neighbors) is plotted to visualize the performance of the model.

5. **Prediction and Visualization**: Finally, a sample image from the testing set is fed into the trained classifier to make a prediction, and the true label along with the predicted label is displayed using Matplotlib.

## Instructions:
To run the code:

1. Make sure you have Python and the required libraries (NumPy, Pandas, scikit-learn, Seaborn, and Matplotlib) installed.
2. Copy the provided code into a Python script or notebook file.
3. Run the script or execute the cells in the notebook.

---

This README file provides an overview of the code functionality, its components, and instructions for running it. It helps users understand the purpose of the code and how to use it effectively.
