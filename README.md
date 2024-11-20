# Machine Learning Project: Regression and Classification Analysis

## Overview

This project explores various **machine learning models** for classification and regression tasks. The main objectives include implementing **Linear Discriminant Analysis (LDA)**, **Quadratic Discriminant Analysis (QDA)**, and **regression techniques** like **Ordinary Least Squares (OLS)** and **Ridge Regression**, as well as comparing their performance under different scenarios.

## Project Structure

```
MLINSIGHTS-DISCRIMINANTSANDREGRESSION/
├── basecode/
│   ├── diabetes.pickle           # Dataset for regression tasks
│   ├── sample.pickle             # Dataset for classification tasks
│   ├── report.pdf                # Final report with results and observations
│   └── script.py                 # Main Python script implementing the models
├── CSE574_project3_description.pdf  # Problem statement and assignment details
└── README.md                     # Project overview and usage instructions

```

## Implemented Problems

### Problem 1: LDA and QDA Classification
- **Goal**: Compare classification performance using LDA and QDA.
- **Outputs**:
  - Decision boundary plots for both LDA and QDA.
  - Accuracy comparison.
- **Key Results**: LDA outperformed QDA with an accuracy of **97%** on test data.

### Problem 2: OLS Regression
- **Goal**: Compare regression models with and without an intercept term.
- **Outputs**:
  - Mean Squared Error (MSE) values for both models.
- **Key Results**: Including an intercept reduced test MSE, improving the model's performance.

### Problem 3: Ridge Regression
- **Goal**: Optimize the regularization parameter (λ) to balance bias and variance.
- **Outputs**:
  - MSE vs. λ plots for train and test data.
  - Comparison of weight magnitudes between OLS and Ridge Regression.
- **Key Results**: The optimal λ was **0.06**, minimizing test MSE.

### Problem 4: Gradient Descent Ridge Regression
- **Goal**: Compare gradient-based optimization with the closed-form solution.
- **Outputs**:
  - Plots comparing train and test MSE for both approaches.
- **Key Results**: Both methods yielded similar results, with gradient descent being useful for large datasets.

### Problem 5: Non-Linear Regression
- **Goal**: Investigate the effect of polynomial degrees on regression performance.
- **Outputs**:
  - Train and test MSE vs. polynomial degree plots (with and without regularization).
- **Key Results**: Regularization mitigated overfitting, with degrees 2 or 3 achieving optimal test performance.

### Problem 6: Summary and Recommendations
- **Goal**: Summarize results and recommend the best approaches.
- **Outputs**:
  - Recommendations based on classification accuracy and regression MSE.
- **Key Results**: 
  - Use LDA for classification with linearly separable data.
  - For regression, prefer Ridge Regression with regularization for better generalization.

## Instructions to Run

1. Ensure the following dependencies are installed:
   ```bash
   pip install numpy scipy matplotlib
   ```
2. Run the `script.py` file:
   ```bash
   python script.py
   ```
3. Review the results printed to the console and visualized in the plots.


## Acknowledgments
This project was completed as part of a **Machine Learning coursework assignment (CSE 574)**, focusing on foundational concepts of regression and classification.

---

