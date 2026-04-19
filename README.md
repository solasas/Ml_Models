# ML Models Repository

A comprehensive collection of machine learning models implemented from scratch and using scikit-learn. Each model is implemented in a Jupyter notebook with detailed explanations, code examples, and visualizations.

##  Table of Contents

- [Supervised Learning Models](#supervised-learning-models)
- [Unsupervised Learning Models](#unsupervised-learning-models)
- [Neural Networks](#neural-networks)
- [Ensemble Methods](#ensemble-methods)
- [Getting Started](#getting-started)

---

## Supervised Learning Models

### 1. **Linear Regression** (`linear_regression.ipynb`)
- **Purpose**: Predicts continuous numerical values based on input features
- **Algorithm**: Uses ordinary least squares (OLS) to fit a linear relationship
- **Use Cases**: House price prediction, stock price forecasting, temperature estimation
- **Key Concepts**: Cost function, gradient descent, feature scaling

### 2. **Logistic Regression** (`LogisticRegression.ipynb`)
- **Purpose**: Binary and multi-class classification problems
- **Algorithm**: Applies sigmoid function to linear regression for probability outputs
- **Use Cases**: Email spam detection, disease diagnosis, customer churn prediction
- **Key Concepts**: Logistic function, decision boundaries, log-likelihood

### 3. **K-Nearest Neighbors (KNN)** (`knn.ipynb`)
- **Purpose**: Classification and regression using proximity to nearest neighbors
- **Algorithm**: Finds k closest training samples and uses their labels/values
- **Use Cases**: Image recognition, recommendation systems, pattern matching
- **Key Concepts**: Distance metrics (Euclidean, Manhattan), k selection, feature normalization

### 4. **Support Vector Machine (SVM)** (`svm.ipynb`)
- **Purpose**: Classification with maximum margin hyperplane
- **Algorithm**: Finds optimal decision boundary that maximizes margin between classes
- **Use Cases**: Text classification, image classification, bioinformatics
- **Key Concepts**: Support vectors, kernel trick, regularization (C parameter)

---

## Unsupervised Learning Models

### 5. **K-Means Clustering** (`Kmeans.ipynb`)
- **Purpose**: Partitions data into k distinct clusters based on similarity
- **Algorithm**: Iteratively assigns points to nearest centroid and updates centroids
- **Use Cases**: Customer segmentation, image compression, document clustering
- **Key Concepts**: Centroid initialization, convergence criteria, elbow method for k selection

### 6. **ID3 Decision Tree** (`ID3.ipynb`)
- **Purpose**: Builds decision trees for classification using information gain
- **Algorithm**: Recursively splits data on features with highest information gain
- **Use Cases**: Medical diagnosis, credit risk assessment, feature importance analysis
- **Key Concepts**: Information entropy, information gain, Gini impurity, tree pruning

---

## Neural Networks

### 7. **Perceptron** (`perceptron.ipynb`)
- **Purpose**: Simple linear classifier using weighted sum and threshold activation
- **Algorithm**: Updates weights based on classification errors
- **Use Cases**: Binary linear classification problems
- **Key Concepts**: Activation functions, weight updates, learning rate, convergence

### 8. **Artificial Neural Network with Backpropagation** (`annwithbackprop.ipynb`)
- **Purpose**: Multi-layer neural network for complex non-linear problems
- **Algorithm**: Forward propagation followed by backpropagation to compute gradients
- **Use Cases**: Image classification, handwritten digit recognition, pattern recognition
- **Key Concepts**: Hidden layers, activation functions (ReLU, sigmoid), gradient descent, epochs

---

## Ensemble Methods

### 9. **Random Forest** (`random_forest.ipynb`)
- **Purpose**: Ensemble of decision trees for improved accuracy and robustness
- **Algorithm**: Trains multiple trees on random subsets and aggregates predictions
- **Use Cases**: Feature importance analysis, regression, classification
- **Key Concepts**: Bootstrap aggregating (bagging), feature subsampling, out-of-bag error

### 10. **Gradient Boosting** (`gradientBoost.ipynb`)
- **Purpose**: Sequential ensemble that corrects errors of previous models
- **Algorithm**: Builds trees sequentially, each correcting residuals of the previous
- **Use Cases**: Kaggle competitions, regression, classification with high accuracy
- **Key Concepts**: Residual fitting, learning rate, tree depth, loss functions

---

## Getting Started

### Prerequisites
```bash
pip install jupyter numpy pandas scikit-learn matplotlib seaborn scipy
