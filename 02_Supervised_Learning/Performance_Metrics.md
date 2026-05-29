# Machine Learning Performance Metrics Cheat Sheet

Evaluating a Machine Learning model is just as important as training it. If you use the wrong metric, you might deploy a model that performs terribly in the real world. 

This guide breaks down the most critical metrics for both **Regression** (predicting a continuous number) and **Classification** (predicting a category), complete with mathematical formulas, plain-English concepts, and real-world examples.

---

## 📈 1. Regression Metrics
*Regression algorithms (like Linear Regression or Random Forest Regressor) predict a continuous value, like the price of a house or tomorrow's temperature. We evaluate them based on how far off their predictions are from the actual values.*

### 1.1 Mean Absolute Error (MAE)
**The Concept:** MAE is the simplest evaluation metric. It calculates the absolute difference between the actual value and the predicted value for every data point, and then takes the average.
**The Formula:**
$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$
*(Where $n$ is the number of samples, $y_i$ is the true value, and $\hat{y}_i$ is the predicted value)*

**Example:** You predict three houses will sell for `$100k`, `$200k`, and `$300k`. They actually sell for `$110k`, `$190k`, and `$300k`. 
Your errors are `$10k`, `$10k`, and `$0`. 
Your MAE is **`$6.66k`**. On average, your model is off by $6.66k.

### 1.2 Mean Squared Error (MSE)
**The Concept:** MSE is similar to MAE, but it **squares** the errors before averaging them. This is crucial because squaring heavily punishes large errors. If your model is off by 10, the squared error is 100. If it's off by 100, the squared error is 10,000!
**The Formula:**
$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Example:** Using the houses above, the errors are $10k, $10k, and 0. 
Squared errors: $100M, $100M, $0. 
MSE = **`66.6 Million`**. (Notice how the unit is now "dollars squared", which is hard to interpret).

### 1.3 Root Mean Squared Error (RMSE)
**The Concept:** Because MSE squares the units (e.g., dollars squared), it's hard to explain to a business stakeholder. RMSE simply takes the square root of the MSE to bring the metric back to the original units. It still maintains the benefit of heavily punishing large errors.
**The Formula:**
$$ RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

**Example:** $\sqrt{66.6\text{M}} \approx$ **`$8.16k`**. (Notice this is higher than the MAE of $6.66k, because RMSE gives more weight to larger errors).

### 1.4 R-Squared ($R^2$) - Coefficient of Determination
**The Concept:** Unlike the error metrics above (where lower is better), $R^2$ is a percentage score from 0.0 to 1.0 (where higher is better). It tells you what percentage of the variance in the target variable is explained by your features. 
**The Formula:**
$$ R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
*(Where $RSS$ is the Residual Sum of Squares (your model's error) and $TSS$ is the Total Sum of Squares (the error if you just guessed the average every time).*

**Example:** If you are predicting house prices based on Square Footage, an $R^2$ of **`0.85`** means that 85% of the differences in house prices can be explained by their Square Footage. The remaining 15% is due to factors you don't have data for (like neighborhood or school ratings).

---

## 🎯 2. Classification Metrics
*Classification algorithms (like Logistic Regression or SVM) predict discrete categories, like "Spam" or "Not Spam". We evaluate them based on how many times they guessed the right category, but the details matter heavily.*

### 2.1 The Confusion Matrix
Before understanding the metrics, you must understand the Confusion Matrix. It maps out exactly how your model got confused. Let's use a **Medical Test for a Disease (1 = Sick, 0 = Healthy)** as an example:

| | Predicted: 0 (Healthy) | Predicted: 1 (Sick) |
|---|---|---|
| **Actual: 0 (Healthy)** | **True Negative (TN)**: Actually healthy, predicted healthy. | **False Positive (FP)**: Actually healthy, predicted sick. *(Type I Error - False Alarm)* |
| **Actual: 1 (Sick)** | **False Negative (FN)**: Actually sick, predicted healthy. *(Type II Error - Dangerous!)* | **True Positive (TP)**: Actually sick, predicted sick. |

### 2.2 Accuracy
**The Concept:** The most intuitive metric. Out of all the predictions we made, what percentage were correct?
**The Formula:**
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

**The Danger:** Accuracy is a terrible metric for **imbalanced datasets**. If 99% of emails are normal and 1% are spam, a broken model that just guesses "Normal" every single time will have 99% Accuracy! But it's completely useless as a spam filter.

### 2.3 Precision
**The Concept:** Out of all the times the model *claimed* something was positive, how many times was it actually right? Precision is about quality. 
**The Formula:**
$$ Precision = \frac{TP}{TP + FP} $$

**Example (Spam Filter):** If your filter flags 100 emails as Spam, but only 80 were actually spam (20 were important work emails), your precision is **`80%`**. You want high precision when False Positives are very costly (you don't want your boss's email going to the spam folder!).

### 2.4 Recall (Sensitivity / True Positive Rate)
**The Concept:** Out of all the *actual* positive cases in the dataset, how many did our model successfully find? Recall is about quantity/coverage.
**The Formula:**
$$ Recall = \frac{TP}{TP + FN} $$

**Example (Cancer Detection):** If 100 patients actually have cancer, and your AI flags 90 of them (missing 10), your recall is **`90%`**. You want high recall when False Negatives are deadly (it is better to falsely alarm a healthy patient than to tell a dying patient they are fine).

### 2.5 The F1-Score
**The Concept:** You often have to trade Precision for Recall. If you want 100% recall on cancer, just guess everyone has cancer! (Your precision will be terrible). The F1-Score is the **Harmonic Mean** of Precision and Recall. It gives you a single score that balances both.
**The Formula:**
$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

**Example:** If your model is great at finding sick people (Recall = 0.95) but causes a lot of false alarms (Precision = 0.40), the F1 score will heavily penalize the model, resulting in an F1-Score of roughly **`0.56`**.

### 2.6 ROC Curve and AUC (Area Under the Curve)
**The Concept:** Many classifiers (like Logistic Regression) output a probability (e.g., "70% chance of being Spam"). By default, we say $\ge 50\%$ is Spam. But what if we change the threshold to $80\%$? 
The **ROC (Receiver Operating Characteristic) Curve** plots the True Positive Rate (Recall) against the False Positive Rate at every possible threshold.

The **AUC (Area Under the Curve)** is a single number summarizing the ROC curve.
- **AUC = 1.0**: A perfect model. It separates the classes flawlessly.
- **AUC = 0.5**: A useless model. It is no better than flipping a coin.
