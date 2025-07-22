<div align="center">
  <a href="https://scikit-learn.org/stable/">
    <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Scikit-learn Logo" width="120"/>
  </a>
</div>

<h1 align="center">Mastering Scikit-learn for Machine Learning</h1>

<p align="center">
  A practical guide to implementing machine learning models using Scikit-learn. This document covers the end-to-end workflow, from data preprocessing and model training to evaluation.
</p>

<p align="center">
  <a href="https://github.com/your-username/your-repo-name/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0%2B-informational.svg" alt="Scikit-learn Version">
</p>

---

This repository culminates the data science journey by focusing on Scikit-learn, the premier library for classical machine learning in Python. Building on the foundations of NumPy, pandas, and Seaborn, this guide provides a clear path to applying predictive modeling techniques to real-world problems.

---

## 📋 Table of Contents

1. [Why Scikit-learn? The Unified ML Toolkit](#why-scikit-learn-the-unified-ml-toolkit)
2. [Installation](#installation)
3. [The Core API: A Consistent Workflow](#the-core-api-a-consistent-workflow)
4. [End-to-End Machine Learning Workflow](#end-to-end-machine-learning-workflow)
5. [Key Modules for Every Project](#key-modules-for-every-project)
6. [Scikit-learn Cheat Sheet](#scikit-learn-cheat-sheet)
7. [Practice Questions](#practice-questions)
8. [Resources](#resources)
9. [Author](#author)

---

## 1. Why Scikit-learn? The Unified ML Toolkit

While deep learning frameworks like TensorFlow and PyTorch dominate complex tasks, Scikit-learn is the undisputed king of traditional machine learning. It provides a vast array of efficient tools for data mining and data analysis built on top of NumPy, SciPy, and Matplotlib.

**Key Idea:** Scikit-learn's greatest strength is its simple, consistent, and unified API. Every algorithm in the library is accessed through the same interface, making it incredibly easy to swap out one model for another and find the best one for your problem.

---

## 2. Installation

Install Scikit-learn using pip:

```bash
pip install scikit-learn
```

Import the library in your Python scripts:

```python
import sklearn
import pandas as pd
import numpy as np
```

---

## 3. The Core API: A Consistent Workflow

Almost every "estimator" (model) in Scikit-learn follows the same pattern:

1. **Choose a model by importing its class**
    ```python
    from sklearn.linear_model import LogisticRegression
    ```
2. **Instantiate the model, set hyperparameters**
    ```python
    model = LogisticRegression(C=1.0)
    ```
3. **Train the model using `.fit()`**
    ```python
    model.fit(X_train, y_train)
    ```
4. **Make predictions using `.predict()`**
    ```python
    predictions = model.predict(X_test)
    ```

This fit/predict pattern is the heart of Scikit-learn.

---

## 4. End-to-End Machine Learning Workflow

Here is the standard process for a supervised learning task:

### Step 1: Choose a Model

Select a model appropriate for your task (e.g., regression or classification).

```python
from sklearn.neighbors import KNeighborsClassifier
```

### Step 2: Prepare Your Data

Split your data into features (X) and target variable (y), then into training and testing sets.

```python
from sklearn.model_selection import train_test_split

# Assume 'df' is your pandas DataFrame
X = df[['feature1', 'feature2', 'feature3']]
y = df['target_variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step 3: Train the Model (.fit())

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

### Step 4: Make Predictions (.predict())

```python
predictions = knn.predict(X_test)
```

### Step 5: Evaluate the Model

Compare predictions to actual values using appropriate metrics.

```python
from sklearn.metrics import accuracy_score, classification_report

print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))
```

---

## 5. Key Modules for Every Project

### Data Preprocessing (`sklearn.preprocessing`)

- **StandardScaler:** Scales features to have zero mean and unit variance.
- **MinMaxScaler:** Scales features to a given range, usually [0, 1].
- **OneHotEncoder:** Converts categorical string features into numerical format.

### Supervised Learning Models

**Classification:**
- `LogisticRegression` (linear_model): Binary classification.
- `KNeighborsClassifier` (neighbors): Distance-based classification.
- `SVC` (svm): Support vector classifier.
- `DecisionTreeClassifier`, `RandomForestClassifier` (tree, ensemble): Tree-based models.

**Regression:**
- `LinearRegression` (linear_model): Classic linear model.
- `Ridge`, `Lasso` (linear_model): Linear models with regularization.

### Unsupervised Learning Models (`sklearn.cluster`, `sklearn.decomposition`)

- **KMeans:** Clustering data into k groups.
- **PCA:** Dimensionality reduction.

### Model Evaluation Metrics (`sklearn.metrics`)

**Classification:**
- `accuracy_score`: Percentage of correct predictions.
- `confusion_matrix`: Table of prediction errors.
- `classification_report`: Precision, recall, F1-score summary.

**Regression:**
- `mean_squared_error`: Average squared difference between predicted and actual values.
- `r2_score`: Coefficient of determination.

---

## 6. 🚀 Scikit-learn Cheat Sheet for Applied ML

<details>
<summary>Click to expand the cheat sheet</summary>

| Module                | Class / Function         | Use Case                                      |
|-----------------------|-------------------------|-----------------------------------------------|
| sklearn.preprocessing | StandardScaler()        | Feature scaling (Z-score normalization)       |
| sklearn.preprocessing | MinMaxScaler()          | Scale features to a [0, 1] range              |
| sklearn.preprocessing | OneHotEncoder()         | Convert categorical features to numeric        |
| sklearn.model_selection | train_test_split()    | Split data into training and testing sets      |
| sklearn.model_selection | GridSearchCV()        | Hyperparameter tuning                         |
| sklearn.linear_model  | LogisticRegression()    | Binary classification                         |
| sklearn.neighbors     | KNeighborsClassifier()  | Classification by nearest neighbors           |
| sklearn.svm           | SVC()                   | Support vector classifier                     |
| sklearn.tree          | DecisionTreeClassifier()| Tree-based classification                     |
| sklearn.ensemble      | RandomForestClassifier()| Ensemble of decision trees                    |
| sklearn.linear_model  | LinearRegression()      | Ordinary least squares regression             |
| sklearn.linear_model  | Ridge(), Lasso()        | Regularized regression                        |
| sklearn.cluster       | KMeans()                | Clustering                                    |
| sklearn.decomposition | PCA()                   | Dimensionality reduction                      |
| sklearn.metrics       | accuracy_score()        | Classification accuracy                       |
| sklearn.metrics       | confusion_matrix()      | Classification error table                    |
| sklearn.metrics       | classification_report() | Precision, recall, F1-score summary           |
| sklearn.metrics       | mean_squared_error()    | Regression error (MSE)                        |
| sklearn.metrics       | r2_score()              | Regression coefficient of determination       |

</details>

---

## 7. Practice Questions

1. Load the Iris dataset and split it into train/test sets.
2. Standardize the features and train a Logistic Regression model.
3. Evaluate the model using accuracy and classification report.
4. Use GridSearchCV to tune hyperparameters for a Random Forest.
5. Build a pipeline that scales data and fits a Support Vector Machine.
6. Perform k-means clustering on the Wine dataset and visualize clusters.
7. Use PCA for dimensionality reduction and plot the first two components.
8. Implement cross-validation for a Decision Tree classifier.
9. Encode categorical features using OneHotEncoder.
10. Save and load a trained model using joblib.

---

## 📚 Resources

- [Scikit-learn Official Documentation (Best for Reference)](https://scikit-learn.org/stable/user_guide.html)  
  The official Scikit-learn user guide and API reference.

- [Video Tutorial (Best for Visual Learners)](https://www.youtube.com/watch?v=0Lt9w-BxKFQ)  
  FreeCodeCamp's "Scikit-learn Course for Beginners" covers all major algorithms and workflows.

---

## Author

**Prakash Sahoo**  
**Email:** prakash2004sahoo@gmail.com
**Linkdine:** https://www.linkedin.com/in/prakash-sahoo-ai/
