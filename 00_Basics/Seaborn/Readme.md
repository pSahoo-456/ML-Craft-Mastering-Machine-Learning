<div align="center">
  <img src="https://seaborn.pydata.org/_static/logo-wide-light.svg" alt="Seaborn Logo" width="400"/>
</div>

<h1 align="center">Mastering Seaborn for Statistical Visualization</h1>

<p align="center">
  A guide to creating beautiful and informative statistical graphics in Python. This document covers the core plotting functions in Seaborn, focusing on their application in exploratory data analysis for machine learning.
</p>

<p align="center">
  <a href="https://github.com/your-username/your-repo-name/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Seaborn-0.11%2B-informational.svg" alt="Seaborn Version">
  </a>
</p>

---

This repository builds upon the data wrangling skills from pandas to explore the world of data visualization with Seaborn. Effective visualization is critical for understanding data distributions, relationships, and patterns before feeding data into a machine learning model.

---

## 📋 Table of Contents

1. [Why Seaborn? The Art of Statistical Storytelling](#why-seaborn-the-art-of-statistical-storytelling)
2. [Installation](#installation)
3. [The Core Idea: A High-Level Interface](#the-core-idea-a-high-level-interface)
4. [Key Plot Categories for Data Analysis](#key-plot-categories-for-data-analysis)
    - Relational Plots: scatterplot & lineplot
    - Categorical Plots: countplot, boxplot, violinplot
    - Distribution Plots: histplot, kdeplot, displot
    - Matrix Plots: heatmap, clustermap
5. [Customizing Plots: Style and Aesthetics](#customizing-plots-style-and-aesthetics)
6. [Seaborn Cheat Sheet](#seaborn-cheat-sheet)
7. [Practice Questions](#practice-questions)
8. [Resources](#resources)
9. [Author](#author)

---

## 1. Why Seaborn? The Art of Statistical Storytelling

Matplotlib is the foundational plotting library in Python, giving you full control over every element of a plot. However, this control comes at the cost of verbosity. Creating a complex, statistically-aware, and aesthetically pleasing plot can require a lot of code.

Seaborn provides a high-level API on top of Matplotlib. It excels at creating sophisticated statistical plots with very little code. It integrates seamlessly with pandas DataFrames, allowing you to create beautiful visualizations directly from your data.

**Key Idea:** Use Seaborn for rapid exploratory data analysis (EDA). Its primary goal is to make it easy to understand the relationships and distributions within your data.

---

## 2. Installation

To get started, install Seaborn using pip. This will also install its dependencies, including Matplotlib and pandas.

```bash
pip install seaborn
```

Then, import it into your Python scripts with its standard alias, `sns`.

```python
import seaborn as sns
import matplotlib.pyplot as plt # For plot customization
import pandas as pd
```

---

## 3. The Core Idea: A High-Level Interface

The power of Seaborn comes from its "dataset-oriented" API. Instead of thinking about "what to plot on x" and "what to plot on y," you think about "which variables in my DataFrame do I want to see the relationship between?"

You pass the entire DataFrame to the plotting function and specify the column names for the x, y, and hue (color) parameters.

```python
# Load a sample dataset from Seaborn's library
tips = sns.load_dataset("tips")

# Create a scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day")
plt.show()
```

---

## 4. Key Plot Categories for Data Analysis

### Relational Plots

Used to understand the relationship between two numerical variables.

- `sns.scatterplot()`: The most fundamental way to see how two variables relate.
- `sns.lineplot()`: Ideal for showing the trend of a variable over time.

### Categorical Plots

Used to visualize the relationship between a numerical and a categorical variable.

- `sns.countplot()`: Shows the counts of observations in each categorical bin.
- `sns.boxplot()`: Visualizes the distribution and outliers of a numerical variable for different categories.
- `sns.violinplot()`: Combines a boxplot with a kernel density estimation (KDE) to show the distribution shape.
- `sns.barplot()`: Shows an estimate of central tendency for a numeric variable for each category.

```python
# Example of a boxplot
sns.boxplot(data=tips, x="day", y="total_bill")
plt.show()
```

### Distribution Plots

Used to understand the distribution of a single numerical variable.

- `sns.histplot()`: A classic histogram showing frequency bins.
- `sns.kdeplot()`: Visualizes the probability density of a continuous variable.
- `sns.displot()`: A figure-level function that can draw histograms, KDEs, and more.

```python
# Example of a histogram with a KDE overlay
sns.histplot(data=tips, x="total_bill", kde=True)
plt.show()
```

### Matrix Plots

Used to visualize matrix-like data, such as correlation matrices.

- `sns.heatmap()`: Creates a color-encoded matrix, perfect for showing correlations between variables.
- `sns.clustermap()`: Hierarchically clusters and plots a heatmap.

```python
# Example of a heatmap for a correlation matrix
numeric_cols = tips.select_dtypes(include=['number'])
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
```

---

## 5. Customizing Plots: Style and Aesthetics

Seaborn makes it easy to create professional-looking plots out of the box.

```python
# Set a theme for all subsequent plots
sns.set_theme(style="whitegrid", palette="viridis")

# Create a plot (it will automatically use the new theme)
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.show()

# Add titles and labels using Matplotlib's functions
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Tip Amount vs. Total Bill")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.show()
```

---

## 6. Seaborn Cheat Sheet

<details>
<summary>Click to expand the cheat sheet</summary>

| Function | Plot Type | Use Case |
|----------|-----------|----------|
| `import seaborn as sns` | Import | Standard import alias |
| `import matplotlib.pyplot as plt` | Import | For plot customization and display |
| `sns.scatterplot(data, x, y)` | Scatter Plot | Relationship between two numeric variables |
| `sns.lineplot(data, x, y)` | Line Plot | Show trends, often over time |
| `sns.countplot(data, x)` | Count Plot | Counts of a single categorical variable |
| `sns.barplot(data, x, y)` | Bar Plot | Central tendency of a numeric variable by category |
| `sns.boxplot(data, x, y)` | Box Plot | Quartiles, median, and outliers by category |
| `sns.violinplot(data, x, y)` | Violin Plot | Box plot + KDE |
| `sns.stripplot(data, x, y)` | Strip Plot | Scatter plot for categorical data |
| `sns.histplot(data, x)` | Histogram | Frequency distribution in bins |
| `sns.kdeplot(data, x)` | KDE Plot | Estimated probability density |
| `sns.displot(data, x, kind)` | Figure-Level Plot | Flexible interface for histplot/kdeplot |
| `sns.jointplot(data, x, y)` | Joint Plot | Scatter + histograms on axes |
| `sns.heatmap(matrix_data)` | Heatmap | Visualizing correlation matrices |
| `sns.pairplot(data)` | Pair Plot | Grid of scatterplots for all pairs |
| `sns.set_theme(style, palette)` | Set Theme | Style and color palette for all plots |
| `plt.title('My Title')` | Add Title | Use Matplotlib to add a title |
| `plt.xlabel('X Label')` | Add Label | Use Matplotlib to label axes |
| `plt.show()` | Display Plot | Render and show the current figure |

</details>

---

## 7. Practice Questions

**Setup:** For the following questions, use the built-in Titanic dataset from Seaborn. Load it with `titanic = sns.load_dataset('titanic')`.

1. **Survival Rate:** Create a countplot to show how many passengers survived versus how many did not. (The 'survived' column has 0s and 1s).
2. **Survival by Class:** Create a countplot that shows the survival count but broken down by passenger class (`pclass`). Use the `hue` parameter.
3. **Age Distribution:** Create a histplot to show the distribution of passenger ages (`age`). What does this tell you about the passengers?
4. **Fare by Class:** Use a boxplot to visualize the distribution of the fare paid by passengers in each `pclass`. What do you observe about the fares?
5. **Age vs. Fare:** Create a scatterplot to see the relationship between age and fare. Color the points by survival status using the `hue` parameter. Is there any obvious pattern?
6. **Correlation Matrix:** Calculate the correlation matrix for the numerical columns in the Titanic dataset. Use a heatmap to visualize this matrix. Which two variables have the strongest positive correlation?
7. **Pairwise Relationships:** Use pairplot on the Titanic dataset. This will create a grid of plots. What is the most interesting relationship you can find from this grid?
8. **Survival Rate by Sex:** Use a barplot to show the average survival rate (the 'survived' column) for each sex.
9. **Age Distribution by Class and Sex:** Use a violinplot to show the distribution of age for each `pclass`, split by sex.

---

## 📚 Resources

- [Seaborn Official Documentation (Best for Reference)](https://seaborn.pydata.org/tutorial.html)  
  The official Seaborn tutorial and API reference.

- [Video Tutorial (Best for Visual Learners)](https://www.youtube.com/watch?v=GcXcSZ0gQps)  
  FreeCodeCamp's "Seaborn Course for Beginners" covers all major plot types and customization.

---

## Author

**Prakash Sahoo**  
**Email:** prakash2004sahoo@gmail.com

**Linkdine:**https://www.linkedin.com/in/prakash-sahoo-ai/