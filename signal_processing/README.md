# Signal Processing

This directory contains the signal processing pipeline used for ECG and PPG analysis in the cuffless blood pressure estimation project.

## Overview

The signal processing module converts raw biosignals into physiological features that can be used for blood pressure estimation. The pipeline includes signal filtering, peak detection, feature extraction, and data validation.

## Main Components

### ECG Processing

* Baseline drift removal
* 60 Hz power-line noise suppression
* 0.5–40 Hz Butterworth band-pass filtering
* R-peak detection based on a Pan–Tompkins-inspired algorithm

### PPG Processing

* 0.5–7 Hz band-pass filtering
* VPG (Velocity Photoplethysmogram) computation
* APG (Acceleration Photoplethysmogram) computation
* Foot-point detection using slope-based analysis

### Feature Extraction

* Pulse Arrival Time (PAT)
* Pulse Transit Time (PTT)
* RR Interval
* Beat-level feature generation

### Data Validation

* Physiological range checks
* Outlier detection
* Beat-to-beat consistency verification

## Processing Pipeline

CSV Input

→ Signal Segmentation

→ ECG/PPG Filtering

→ ECG R-Peak Detection

→ PPG Foot Detection

→ PAT/PTT Calculation

→ Data Validation

→ Feature Export

## Output Files

* Filtered ECG and PPG signals
* R-peak detection results
* PPG foot detection results
* Beat-level feature tables
* Summary statistics
* Visualization plots

## Technologies

* Python
* NumPy
* SciPy
* Pandas
* Matplotlib

