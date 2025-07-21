<div align="center">
  <img src="https://pandas.pydata.org/docs/_static/pandas.svg" alt="Pandas Logo" width="400"/>
</div>

<h1 align="center">Mastering Pandas for AI/ML</h1>

<p align="center">
  A comprehensive, code-first guide to mastering the Pandas library, focusing on the core concepts essential for Artificial Intelligence and Machine Learning applications.
</p>

<p align="center">
  <a href="https://github.com/pSahoo-456/ML-Craft-Mastering-Machine-Learning/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Pandas-1.3%2B-informational.svg" alt="Pandas Version">
</p>

---

## Table of Contents

1. [Why Pandas?](#why-pandas)
2. [Installation](#installation)
3. [Core Data Structures](#core-data-structures)
4. [Essential Operations](#essential-operations)
    - [Selection & Indexing](#selection--indexing)
    - [Filtering & Conditional Selection](#filtering--conditional-selection)
    - [Handling Missing Data](#handling-missing-data)
    - [Grouping & Aggregation](#grouping--aggregation)
    - [Merging & Concatenation](#merging--concatenation)
    - [Applying Functions](#applying-functions)
5. [Pandas Cheat Sheet](#pandas-cheat-sheet)
6. [Practice Questions](#practice-questions)
7. [Resources](#resources)

---

## 1. Why Pandas?

Pandas is the go-to library for data manipulation and analysis in Python. It provides powerful, flexible, and easy-to-use data structures for working with structured data, making it indispensable for AI/ML workflows.

- **Python Lists/Dicts:** Good for simple data, but lack structure and speed.
- **Pandas DataFrame:** Tabular, labeled, fast, and feature-rich. Enables efficient data cleaning, exploration, and transformation.

---

## 2. Installation

Install Pandas using pip:

```bash
pip install pandas
```

Import it into your Python scripts:

```python
import pandas as pd
```

---

## 3. Core Data Structures

**Series:** 1D labeled array  
**DataFrame:** 2D labeled table (rows & columns)

```python
# Series
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'Chicago']
})
```

---

## 4. Essential Operations

### Selection & Indexing

```python
# Select a column
ages = df['Age']

# Select multiple columns
subset = df[['Name', 'City']]

# Select rows by label
row = df.loc[1]

# Select rows by position
row = df.iloc[1]
```

### Filtering & Conditional Selection

```python
# Filter rows where Age > 30
df[df['Age'] > 30]

# Filter by multiple conditions
df[(df['Age'] > 25) & (df['City'] == 'NY')]
```

### Handling Missing Data

```python
# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df.dropna()

# Fill missing values
df.fillna({'Age': df['Age'].mean(), 'City': 'Unknown'})
```

### Grouping & Aggregation

```python
# Group by column and aggregate
df.groupby('City')['Age'].mean()

# Count unique values
df['City'].value_counts()
```

### Merging & Concatenation

```python
# Concatenate DataFrames
pd.concat([df1, df2])

# Merge DataFrames (SQL-style join)
pd.merge(df1, df2, on='key')
```

### Applying Functions

```python
# Apply function to column
df['Age Category'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')
```

---

## 5. Pandas Cheat Sheet

<details>
<summary>Click to expand the cheat sheet</summary>

### Syntax | Description

**Importing**

```python
import pandas as pd
```

**Data Structures**

```python
pd.Series(data, index=idx)         # 1D labeled array
pd.DataFrame(dict)                 # 2D labeled data structure
```

**I/O**

```python
pd.read_csv('file.csv')            # Read CSV
df.to_csv('new_file.csv')          # Write CSV
```

**Inspection**

```python
df.head(n)                         # First n rows
df.tail(n)                         # Last n rows
df.info()                          # Summary
df.describe()                      # Stats
df.shape                           # Dimensions
df.columns                         # Column labels
df['col'].value_counts()           # Unique counts
```

**Selection**

```python
df['col']                          # Single column
df[['col1', 'col2']]               # Multiple columns
df.loc[label]                      # Row by label
df.iloc[pos]                       # Row by position
```

**Filtering**

```python
df[df['col'] > 0]                  # Condition
df[df['col'].isin([val1, val2])]   # List of values
```

**Data Cleaning**

```python
df.isnull().sum()                  # Missing values
df.dropna()                        # Drop missing
df.fillna(value)                   # Fill missing
df.drop_duplicates()               # Remove duplicates
```

**Grouping & Aggregating**

```python
df.groupby('col').sum()            # Aggregate after group
```

**Combining**

```python
pd.concat([df1, df2])              # Concatenate
pd.merge(df1, df2, on='key')       # Merge
```

**Applying Functions**

```python
df['col'].apply(func)              # Apply function
```

</details>


## 📚 Resources

- [Pandas Official Documentation (Best for Reference)](https://pandas.pydata.org/docs/user_guide/index.html)  
  The official Pandas user guide is the most reliable and comprehensive reference for all features and functions.

- [Video Tutorial (Best for Visual Learners)](https://www.youtube.com/watch?v=vmEHCJofslg)  
  FreeCodeCamp's "Pandas Course for Beginners" provides a thorough, beginner-friendly video walkthrough of Pandas basics and practical