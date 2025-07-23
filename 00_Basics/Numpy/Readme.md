<div align="center">
  <img src="https://numpy.org/images/logo.svg" alt="NumPy Logo" width="400"/>
</div>

<h1 align="center">Mastering NumPy for AI/ML</h1>

<p align="center">
  A comprehensive, code-first guide to mastering the NumPy library, focusing on the core concepts essential for Artificial Intelligence and Machine Learning applications.
</p>

<p align="center">
  <a href="https://github.com/pSahoo-456/ML-Craft-Mastering-Machine-Learning/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/NumPy-1.21%2B-informational.svg" alt="NumPy Version">
</p>

---

## Table of Contents
0. [History of NumPy](#history-of-numpy)
1. [Why NumPy?](#why-numpy)
2. [Installation](#installation)
3. [The ndarray: Creation & Inspection](#the-ndarray-creation--inspection)
4. [Core Concepts for AI/ML](#core-concepts-for-aiml)
5. [Data Manipulation](#data-manipulation)
    - [Indexing and Slicing](#indexing-and-slicing)
    - [Boolean Indexing (Filtering)](#boolean-indexing-filtering)
    - [Aggregations & the axis Parameter](#aggregations--the-axis-parameter)
6. [Linear Algebra: The Language of ML](#linear-algebra-the-language-of-ml)
7. [Other Essential Operations](#other-essential-operations)
    - [Reshaping & Resizing](#reshaping--resizing)
    - [Stacking & Splitting](#stacking--splitting)
8. [NumPy Cheat Sheet](#numpy-cheat-sheet)
9. [Practice Questions](#practice-questions)

---

## 0. History of NumPy

NumPy originated as Numeric in the mid-1990s, created by Jim Hugunin to bring efficient array computing to Python. In 2005, **Travis Oliphant** merged Numeric with another array package, Numarray, to create NumPy as we know it today. Since then, NumPy has become the foundational library for scientific computing in Python, powering libraries like SciPy, pandas, scikit-learn, and many more. Its efficient n-dimensional array object and broadcasting capabilities have made it the standard for numerical computation in the Python ecosystem.

---


## 1. Why NumPy?

At its heart, Artificial Intelligence and Machine Learning are about finding patterns in numerical data. NumPy (Numerical Python) is the most fundamental package for high-performance numerical computation in Python. It is the bedrock upon which libraries like pandas, Scikit-learn, TensorFlow, and PyTorch are built.

- **Python Lists:** Flexible, can hold different data types. This flexibility makes them slow.
- **NumPy ndarray:** A grid of values of a single type. Stored in contiguous memory, enabling massive speedups.

For any serious data work, NumPy is non-negotiable for its speed and efficiency.

---

## 2. Installation

Install NumPy using pip:

```bash
pip install numpy
```

Import it into your Python scripts:

```python
import numpy as np
```

---

## 3. The ndarray: Creation & Inspection

The core object in NumPy is the n-dimensional array, or `ndarray`.

**Creating Arrays:**

```python
arr_1d = np.array([1, 2, 3, 4, 5])                # 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])         # 2D array (matrix)
zeros = np.zeros((3, 4))                          # 3x4 matrix of zeros
ones = np.ones((2, 5))                            # 2x5 matrix of ones
seq = np.arange(0, 10, 2)                         # [0, 2, 4, 6, 8]
points = np.linspace(0, 1, 5)                     # 5 points from 0 to 1
rand_int = np.random.randint(0, 100, (3, 3))      # 3x3 matrix of random ints
```

**Inspecting Array Attributes:**

```python
print(arr_2d.shape)   # (2, 3)
print(arr_2d.ndim)    # 2
print(arr_2d.size)    # 6
print(arr_2d.dtype)   # int64
```

---

## 4. Core Concepts for AI/ML

### Vectorization

Perform operations on entire arrays at once, instead of looping.

```python
prices = np.array([100, 150, 200, 220])
taxed_prices = prices * 1.10
# Output: array([110. , 165. , 220. , 242. ])
```

### Broadcasting

NumPy "stretches" smaller arrays to match shapes during arithmetic operations.

```python
inputs = np.zeros((3, 4))           # Batch of 3 inputs, 4 features each
bias = np.array([1, 2, 3, 4])       # Bias vector for 4 features
inputs_with_bias = inputs + bias
# [[1. 2. 3. 4.]
#  [1. 2. 3. 4.]
#  [1. 2. 3. 4.]]
```

---

## 5. Data Manipulation

### Indexing and Slicing

```python
data = np.arange(1, 10).reshape(3, 3)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

element = data[2, 1]      # 8
col_2 = data[:, 1]        # array([2, 5, 8])
sub_matrix = data[:2, :2] # [[1 2], [4 5]]
```

### Boolean Indexing (Filtering)

Select data based on conditions.

```python
grades = np.array([75, 88, 92, 64, 99])
passing_grades = grades[grades >= 75] # array([75, 88, 92, 99])
```

### Aggregations & the axis Parameter

Summarize data with functions like `np.sum`, `np.mean`, `np.std`.

- `axis=0`: Down the columns (collapses rows)
- `axis=1`: Across the rows (collapses columns)

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
col_sums = np.sum(matrix, axis=0)  # array([5, 7, 9])
row_sums = np.sum(matrix, axis=1)  # array([6, 15])
```

---

## 6. Linear Algebra: The Language of ML

Matrix multiplication is performed with the `@` operator.

```python
weights = np.array([[0.1, 0.2], [-0.5, 0.7], [0.3, 0.1]]) # (3, 2)
inputs = np.random.rand(4, 3)                             # (4, 3)
output = inputs @ weights                                 # (4, 2)

print(inputs.shape)   # (4, 3)
print(weights.shape)  # (3, 2)
print(output.shape)   # (4, 2)
```

---

## 7. Other Essential Operations

### Reshaping & Resizing

- `reshape()`: Change array shape without changing data.
- `flatten()`: Returns a 1D copy.
- `ravel()`: Returns a 1D "view" (memory efficient).

### Stacking & Splitting

- `np.vstack()`, `np.hstack()`: Stack arrays vertically/horizontally.
- `np.concatenate()`: Join arrays along an axis.
- `np.split()`: Split into sub-arrays.

---

## 8. NumPy Cheat Sheet

<details>
<summary>Click to expand the cheat sheet</summary>

### Syntax | Description

**Importing**

```python
import numpy as np
```
Standard import alias.

**Creating Arrays**

```python
np.array(list)              # From Python list
np.zeros((rows, cols))      # Zeros
np.ones((rows, cols))       # Ones
np.arange(start, stop, step)# Regular step size
np.linspace(start, stop, num)# Specific number of points
np.random.rand(d0, d1)      # Random floats [0, 1)
np.random.randn(d0, d1)     # Standard normal
np.random.randint(low, high, size) # Random integers
```

**Inspecting Arrays**

```python
arr.shape   # Dimensions
arr.dtype   # Data type
arr.ndim    # Number of dimensions
```

**Math & Vectorization**

```python
arr +-*/ 5          # Element-wise with scalars
arr_a +-*/ arr_b    # Element-wise between arrays
np.sqrt(arr), np.log(arr) # Universal Functions (UFuncs)
```

**Slicing & Indexing**

```python
arr[5]          # 1D element
arr[2, 3]       # 2D element
arr[0, :]       # First row
arr[:, 0]       # First column
arr[arr > 50]   # Boolean indexing
```

**Aggregations**

```python
np.sum(arr), np.mean(arr)      # Whole array
np.sum(arr, axis=0)            # Down columns
np.sum(arr, axis=1)            # Across rows
np.argmax(arr), np.argmin(arr) # Index of max/min
```

**Manipulation**

```python
arr.reshape(rows, cols)        # Change shape
arr.T or arr.transpose()       # Transpose
np.concatenate([arr1, arr2], axis=0) # Join arrays
```

**Linear Algebra**

```python
A @ B               # Matrix multiplication
np.dot(A, B)        # Matrix multiplication
np.linalg.inv(A)    # Inverse of matrix
```

</details>

---
...existing code...

---

## 📚 Resources

- [NumPy Official Documentation (Best for Reference)](https://numpy.org/doc/stable/user/quickstart.html)  
  The official NumPy quickstart tutorial is the most reliable and comprehensive reference for all features and functions.

- [Video Tutorial (Best for Visual Learners)](https://www.youtube.com/watch?v=QUT1VHiLmmI)  
  FreeCodeCamp's "NumPy Course for Beginners" provides a thorough, beginner-friendly video walkthrough of NumPy basics and practical applications.

---


Happy Learning!

---

**Author:** Prakash Sahoo  
**Email:** prakash2004sahoo@gmail.com
**Linkdine:** https://www.linkedin.com/in/prakash-sahoo-ai/