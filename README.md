# Wine Quality Classification Using XGBoost

## Project Overview
This project implements an optimized **XGBoost (Extreme Gradient Boosting)** framework to evaluate and classify red wine samples based on physicochemical properties using the **UCI Wine Quality Dataset**. The lab showcases data splitting, multi-class label preprocessing corrections, baseline boosting modeling, and targeted hyperparameter sweep tuning using Scikit-Learn’s `GridSearchCV`.

## Objectives
* **XGBoost Framework Deployment:** Fit and train an un-tuned gradient boosted tree ensemble classifier.
* **Target Array Normalization:** Implement `LabelEncoder` processing to ensure multiclass target structures comply strictly with XGBoost structural constraints.
* **Hyperparameter Grid Tuning:** Optimize structural parameters (`learning_rate`, `subsample`, `min_child_weight`) using joint cross-validation patterns to curb baseline variance overfitting.

## Technical Pipeline

### 1. Missing Offsets and Target Encoding
The source dataset classifies wine quality categories across discrete ordinal steps ranging between $3$ and $8$. 
XGBoost's classification engine mathematically requires multiclass labels to be structured as consecutive, non-negative integers starting strictly at index $0$ ($0, 1, 2, \dots$). 

To fix this, a **`LabelEncoder`** object transforms target maps before model execution:
* Original Class Boundaries: `[3, 4, 5, 6, 7, 8]`
* Transformed Class Targets: `[0, 1, 2, 3, 4, 5]`

### 2. The Overfitting Challenge (Baseline Model)
The baseline model quickly hits a **100% accuracy score on the training data**, while stalling significantly lower on validation points. This variance gap represents standard overfitting, where individual boosted paths memorize local training noise instead of discovering universal mathematical boundaries.

### 3. Hyperparameter Optimization & Fine-Tuning
To mitigate tree depth dominance and improve model generalization, an cross-validated hyperparameter search was conducted.
