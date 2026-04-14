snake_case, comment, leave adequate whitespace

EMG Hand Movement Classification – Action Plan

This repository contains code and documentation for classifying hand movements using EMG signals from the Ninapro DB1 and DB2 datasets. Below is our structured action plan for the project.

1. Dataset Acquisition
Download Ninapro DB1 or DB2 datasets from the official Ninapro database
.
Organize data by subject and movement type.
Store EMG signals, labels, and any additional metadata (e.g., accelerometer data) in a consistent folder structure.

Folder structure example:

data/
 ├─ DB1/
 │   ├─ S1.mat
 │   ├─ S2.mat
 │   └─ ...
 └─ DB2/
     ├─ S1.mat
     ├─ S2.mat
     └─ ...
2. Preprocessing
Apply bandpass filtering to remove noise (typical: 20–450 Hz).
Rectify signals (absolute value) and normalize per channel.
Segment signals into overlapping windows (e.g., 200 ms window, 50% overlap).
3. Feature Extraction
Extract time-domain features from each window:
Mean Absolute Value (MAV)
Root Mean Square (RMS)
Waveform Length (WL)
Optional: Zero Crossings (ZC), Slope Sign Changes (SSC)
Construct feature vectors for each window:
DB1: 8 channels × 3–5 features → 24–40 dimensions
DB2: 12 channels × 3–5 features → 36–60 dimensions
4. Data Splitting
Split dataset into training and testing sets (e.g., 80% train / 20% test).
Optionally, perform cross-validation to evaluate model robustness.
5. Classification
Train machine learning models on feature vectors:
Random Forest (baseline, easy to implement)
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Optional: Neural Networks / CNN / LSTM for raw EMG inputs
Evaluate model using:
Accuracy
Confusion matrix
Classification report (precision, recall, F1-score)
6. Optimization
Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
Feature selection or dimensionality reduction (e.g., PCA) to improve performance.
Explore window size and overlap parameters for best temporal resolution.
7. Documentation & Reporting
Include:
Dataset description
Preprocessing steps
Feature extraction details
Classifier choices and hyperparameters
Model performance metrics
Optional: visualize EMG signals, feature distributions, and confusion matrices.
8. Future Work / Extensions
Test deep learning approaches (CNN, LSTM) on raw EMG data.
Real-time EMG hand movement classification using a wearable device.
Combine EMG with accelerometer/IMU data for improved accuracy.

Noah - Converting MATLAB data to Python and uploading to Github

Kiran - train AI model

Bader - test AI model

Evan - Powerpoint

Agent AI was used to develop this project
