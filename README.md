# AI & ML Internship Task 4 – Classification with Logistic Regression

## Objective

The objective of this task is to build a binary classification model using Logistic Regression and understand classification evaluation metrics, threshold tuning, and the sigmoid function.

## Dataset

**Breast Cancer Wisconsin Dataset**

The dataset contains measurements of breast cell nuclei and a diagnosis column indicating whether a tumor is:

- `B` – Benign
- `M` – Malignant

The `id` column was removed because it is only an identifier, and the empty `Unnamed: 32` column was removed because it contained no useful data.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Steps Performed

### 1. Data Loading and Inspection

The dataset was loaded using Pandas and inspected for:

- Dataset structure
- Data types
- Missing values

### 2. Data Preprocessing

- Removed the `id` column.
- Removed the completely empty `Unnamed: 32` column.
- Separated the features (`X`) and target (`y`).
- Converted the diagnosis labels:
  - `B → 0`
  - `M → 1`

### 3. Train/Test Split

The dataset was divided into:

- **80% training data:** 455 samples
- **20% testing data:** 114 samples

### 4. Feature Standardization

`StandardScaler` was used to standardize the numerical features so that the features are on a comparable scale.

### 5. Logistic Regression Model

A Logistic Regression model was trained using the standardized training data.

### 6. Predictions

The trained model was used to predict whether test samples were benign or malignant.

- `0` = Benign
- `1` = Malignant

### 7. Confusion Matrix

The confusion matrix obtained was:

```text
[[70  1]
 [ 2 41]]
```

This means:

- True Negatives (TN) = 70
- False Positives (FP) = 1
- False Negatives (FN) = 2
- True Positives (TP) = 41

### 8. Precision and Recall

The model achieved:

- **Precision:** 97.62%
- **Recall:** 95.35%

Precision indicates how many of the samples predicted as malignant were actually malignant.

Recall indicates how many of the actual malignant samples were correctly detected.

### 9. ROC-AUC

The ROC-AUC score was:

**0.9974**

This indicates that the model has excellent ability to distinguish between benign and malignant classes.

### 10. Threshold Tuning

The default classification threshold of 0.5 was changed to 0.3.

At threshold 0.3:

- **Precision:** 91.30%
- **Recall:** 97.67%

Lowering the threshold increased recall because the model became more likely to classify a sample as malignant. However, precision decreased because more false positive predictions were made.

### 11. Sigmoid Function

Logistic Regression uses the sigmoid function to convert a raw model score into a probability between 0 and 1.

The sigmoid function is:

```text
σ(z) = 1 / (1 + e^(-z))
```

The sigmoid curve was plotted and saved as `sigmoid_curve.png`.

The general process is:

```text
Input Features
      ↓
Logistic Regression
      ↓
Raw Score
      ↓
Sigmoid Function
      ↓
Probability (0 to 1)
      ↓
Classification Threshold
      ↓
Benign / Malignant
```

## Results Summary

| Metric           | Result |
| ---------------- | -----: |
| Training Samples |    455 |
| Testing Samples  |    114 |
| Features         |     30 |
| Precision        | 97.62% |
| Recall           | 95.35% |
| ROC-AUC          | 0.9974 |
| Threshold        |    0.5 |
| Tuned Threshold  |    0.3 |
| Tuned Precision  | 91.30% |
| Tuned Recall     | 97.67% |

## Interview Questions

### 1. How does Logistic Regression differ from Linear Regression?

Linear Regression is mainly used to predict continuous numerical values, such as house prices.

Logistic Regression is used for classification. It predicts the probability of a class and can be used to classify data into categories such as 0 or 1.

### 2. What is the Sigmoid Function?

The sigmoid function is an S-shaped mathematical function that converts a raw value into a value between 0 and 1. Logistic Regression uses it to obtain probabilities.

### 3. What is Precision vs Recall?

**Precision** tells us how many of the samples predicted as positive are actually positive.

**Recall** tells us how many of the actual positive samples were correctly identified.

### 4. What is the ROC-AUC Curve?

The ROC curve shows how the model's true positive rate and false positive rate change at different classification thresholds.

AUC represents the overall ability of the model to distinguish between the two classes. A value closer to 1 indicates better classification performance.

### 5. What is a Confusion Matrix?

A confusion matrix is a table that compares actual classes with predicted classes.

It contains:

- True Positive
- True Negative
- False Positive
- False Negative

### 6. What happens if classes are imbalanced?

If one class has many more samples than another, the model may become biased toward the majority class. Accuracy alone may then be misleading.

Metrics such as precision, recall, F1-score, and ROC-AUC can provide a better evaluation.

### 7. How do you choose the threshold?

The default threshold is usually 0.5. However, it can be changed depending on the application.

A lower threshold generally increases recall, while a higher threshold generally increases precision.

The choice depends on whether missing positive cases or generating false alarms is more costly.

### 8. Can Logistic Regression be used for multi-class problems?

Yes. Logistic Regression can be extended to multi-class classification using approaches such as One-vs-Rest or multinomial Logistic Regression.

## Files

```text
Task-04-Logistic-Regression/
│
├── dataset/
│   └── data.csv
│
├── logistic_regression.py
│
├── sigmoid_curve.png
│
└── README.md
```

## Conclusion

A Logistic Regression model was successfully implemented for binary classification using the Breast Cancer Wisconsin Dataset. The model achieved high precision, recall, and ROC-AUC performance. Threshold tuning was also performed to understand the trade-off between precision and recall, and the sigmoid function was visualized to understand how Logistic Regression converts model scores into probabilities.
