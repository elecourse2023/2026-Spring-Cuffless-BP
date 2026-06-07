# Validation

This directory contains validation protocols, quality assessment methods, and performance evaluation procedures.

## Objective

The validation module ensures that collected biosignals and machine learning outputs satisfy physiological plausibility and clinical evaluation requirements.

## Experimental Validation

### Subject Criteria

* Healthy adults
* Age range: 20–25 years
* No known cardiovascular disease

### Measurement Conditions

Participants were instructed to:

* Avoid caffeine before measurement
* Avoid smoking before measurement
* Avoid strenuous exercise before measurement
* Remain seated and relaxed during recording

## Signal Quality Assessment (SQI)

Signal quality was evaluated using:

### ECG

* R-peak detection consistency
* Baseline stability
* Signal continuity

### PPG

* Template matching
* Skewness
* Kurtosis
* Zero-crossing rate

## Data Rejection Criteria

Data segments were rejected when:

* Signal clipping occurred
* SQI score < 0.7
* Heart rate < 30 bpm
* Heart rate > 200 bpm
* Severe motion artifacts were detected

## Clinical Evaluation Metrics

Performance was assessed using:

* MAE
* RMSE
* Mean Error
* Standard Deviation
* R² Score

## Clinical Reference Standard

Results were compared against the requirements of:

AAMI / ESH / ISO 81060-2

Clinical acceptance criteria:

* Mean Error ≤ 5 mmHg
* Standard Deviation ≤ 8 mmHg

## Validation Outputs

* Error distribution analysis
* Feature importance analysis
* Model comparison reports
* Subject-independent evaluation results

