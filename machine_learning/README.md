# Machine Learning

This directory contains machine learning models used for cuffless blood pressure estimation.

## Objective

The objective of this module is to estimate systolic blood pressure (SBP) and diastolic blood pressure (DBP) from physiological and demographic features.

## Input Features

* Age
* Gender
* Height
* Weight
* Pulse Arrival Time (PAT)
* Pulse Transit Time (PTT)

## Target Variables

* Systolic Blood Pressure (SBP)
* Diastolic Blood Pressure (DBP)

## Implemented Models

### Support Vector Regression (SVR)

SVR utilizes an RBF kernel to model nonlinear relationships between physiological signals and blood pressure.

Key parameters:

* C
* epsilon
* gamma

### Random Forest (RF)

Random Forest uses an ensemble of decision trees to capture nonlinear interactions among input features.

Key parameters:

* n_estimators
* max_depth
* min_samples_split

### Hybrid LASSO + XGBoost

A two-stage regression framework:

1. LASSO captures global linear trends.
2. XGBoost learns residual nonlinear patterns.

This architecture improves robustness while reducing overfitting.

## Model Evaluation

Performance metrics include:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score
* Mean Error Rate (%)

## Validation Strategy

* Internal Train/Test Split
* Cross Validation
* Leave-One-Group-Out (LOGO) Validation
* Subject-Independent Evaluation

## Technologies

* Scikit-learn
* XGBoost
* NumPy
* Pandas

