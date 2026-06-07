# Hardware

This directory contains hardware-related resources for the ECG/PPG-based cuffless blood pressure estimation system.

## System Overview

The hardware platform simultaneously acquires ECG and PPG signals using a single STM32 microcontroller.

## Components

### ECG Sensor

**AD8232**

Features:

* Single-lead ECG acquisition
* Integrated analog filtering
* Low-power operation

### PPG Sensor

**MAX30102**

Features:

* Infrared and red LED measurement
* Digital optical sensing
* Wearable-friendly form factor

### Microcontroller

**STM32**

Responsibilities:

* ECG ADC sampling
* PPG I2C communication
* Timestamp generation
* USB serial transmission

## System Architecture

AD8232 ECG Sensor

*

MAX30102 PPG Sensor

↓

STM32 Microcontroller

↓

USB Serial Communication

↓

PC Data Logging System

## Data Acquisition

Sampling rate:

* Approximately 250 Hz

Recorded data:

* Timestamp
* Sample Index
* ECG Raw Signal
* Baseline-Removed ECG Signal
* PPG Signal
* Lead-Off Status

## Synchronization Strategy

To minimize timing mismatch between ECG and PPG signals:

* A common STM32 timer was used
* Signals were acquired within a unified acquisition loop
* Timestamps were attached to every sample

## Communication Interfaces

### ADC

Used for:

* ECG acquisition

### I2C

Used for:

* MAX30102 communication

### USB Serial

Used for:

* Real-time data transmission to PC

## Deliverables

* STM32 firmware
* Sensor interface drivers
* Data acquisition framework
* Hardware integration documentation

