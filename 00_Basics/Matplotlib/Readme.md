<div align="center">
  <img src="https://matplotlib.org/_static/logo2_compressed.svg" alt="Matplotlib Logo" width="400"/>
</div>

<h1 align="center">Mastering Matplotlib for Data Science & AI</h1>

<p align="center">
  A practical, code-first guide to mastering the Matplotlib library for data visualization. Learn to create stunning, insightful charts and plots to explore, analyze, and present your data for machine learning and analytics.
</p>

<p align="center">
  <a href="https://github.com/pSahoo-456/ML-Craft-Mastering-Machine-Learning/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Matplotlib-3.4%2B-informational.svg" alt="Matplotlib Version">
</p>

---

## 📋 Table of Contents

1. [Why Matplotlib?](#why-matplotlib)
2. [Installation](#installation)
3. [Basic Plotting](#basic-plotting)
4. [Customizing Plots](#customizing-plots)
5. [Multiple Plots & Subplots](#multiple-plots--subplots)
6. [Visualizing Data for ML](#visualizing-data-for-ml)
7. [Matplotlib Cheat Sheet](#matplotlib-cheat-sheet)
8. [Practice Questions](#practice-questions)
9. [Resources](#resources)

---

## 1. Why Matplotlib?

Matplotlib is the foundational Python library for data visualization. It enables you to create publication-quality plots, charts, and figures with full control over every element. Whether you’re exploring data, building dashboards, or presenting results, Matplotlib is essential for every data scientist and machine learning practitioner.

- **Flexible:** Supports line, bar, scatter, histogram, pie, and more.
- **Customizable:** Control colors, styles, labels, legends, axes, and more.
- **Integrates:** Works seamlessly with NumPy, Pandas, and other libraries.

---

## 2. Installation

Install Matplotlib using pip:

```bash
pip install matplotlib
```

Import it into your Python scripts:

```python
import matplotlib.pyplot as plt
```

---

## 3. Basic Plotting

Create simple line, scatter, and bar plots.

```python
import matplotlib.pyplot as plt

# Line plot
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Simple Line Plot')
plt.show()

# Scatter plot
plt.scatter(x, y, color='red')
plt.title('Scatter Plot')
plt.show()

# Bar plot
plt.bar(x, y, color='green')
plt.title('Bar Plot')
plt.show()
```

---

## 4. Customizing Plots

Change colors, markers, line styles, add legends, and annotate.

```python
plt.plot(x, y, color='purple', marker='o', linestyle='--', label='Data')
plt.legend()
plt.grid(True)
plt.annotate('Peak', xy=(4, 30), xytext=(3, 32),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

---

## 5. Multiple Plots & Subplots

Visualize multiple datasets or chart types in one figure.

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

axs[0, 0].plot(x, y)
axs[0, 0].set_title('Line')

axs[0, 1].scatter(x, y)
axs[0, 1].set_title('Scatter')

axs[1, 0].bar(x, y)
axs[1, 0].set_title('Bar')

axs[1, 1].hist(y)
axs[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

---

## 6. Visualizing Data for ML

Explore and present data distributions, relationships, and model results.

```python
import numpy as np

# Histogram for feature distribution
data = np.random.randn(1000)
plt.hist(data, bins=30, color='skyblue')
plt.title('Feature Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Correlation scatter plot
x = np.random.rand(100)
y = x + np.random.normal(0, 0.1, 100)
plt.scatter(x, y)
plt.title('Feature Correlation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

---

## 7. Matplotlib Cheat Sheet

<details>
<summary>Click to expand the cheat sheet</summary>

### Syntax | Description

**Importing**

```python
import matplotlib.pyplot as plt
```

**Basic Plots**

```python
plt.plot(x, y)                # Line plot
plt.scatter(x, y)             # Scatter plot
plt.bar(x, y)                 # Bar plot
plt.hist(data, bins=20)       # Histogram
plt.pie(sizes, labels=labels) # Pie chart
```

**Customization**

```python
plt.xlabel('Label')           # X axis label
plt.ylabel('Label')           # Y axis label
plt.title('Title')            # Plot title
plt.legend()                  # Show legend
plt.grid(True)                # Show grid
plt.xlim([xmin, xmax])        # X axis limits
plt.ylim([ymin, ymax])        # Y axis limits
```

**Subplots**

```python
fig, axs = plt.subplots(2, 2) # 2x2 grid of plots
axs[0, 0].plot(x, y)
```

**Saving Figures**

```python
plt.savefig('figure.png')     # Save plot to file
```

</details>

---

## 8. Practice Questions

1. **Line Plot:** Plot the accuracy of a model over 20 epochs.
2. **Feature Distribution:** Visualize the distribution of a feature using a histogram.
3. **Correlation:** Create a scatter plot for two features and add a regression line.
4. **Class Balance:** Show the class distribution in a dataset using a bar chart.
5. **Multiple Subplots:** Display four different feature distributions in a 2x2 subplot grid.
6. **Annotations:** Annotate the maximum value in a plot.
7. **Customization:** Change the color, marker, and line style of a plot.
8. **Save Figure:** Save a plot as a PNG file.
9. **Pie Chart:** Visualize the proportion of categories in a dataset.
10. **Confusion Matrix:** Plot a confusion matrix as a heatmap.

---

## 📚 Resources

- [Matplotlib Official Documentation (Best for Reference)](https://matplotlib.org/stable/contents.html)  
  The official Matplotlib documentation covers all features, customization, and examples.

- [Video Tutorial (Best for Visual Learners)](https://www.youtube.com/watch?v=3Xc3CA655Y4)  
  FreeCodeCamp's "Matplotlib Course for Beginners" provides a comprehensive, beginner-friendly video walkthrough.

---

Happy Visualizing! 🎨📊

---
**Author:** Prakash Sahoo  
**Email:** prakash2004sahoo@gmail.com