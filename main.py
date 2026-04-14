"""
=============================================================================
EMG GESTURE CLASSIFICATION PIPELINE
=============================================================================
This script classifies hand movements from surface electromyography (EMG)
signals recorded from the forearm. The goal is to train a machine learning
model that can predict which of 52 hand movements a person is performing,
based solely on muscle activity for use in bionic hand prosthetic technology.

Dataset: NinaPro DB1
  - 27 intact subjects
  - 52 hand movements split across 3 exercises
  - 12 restimuli in Exercise A, 17 restimuli in Exercise B,
    23 restimuli in Exercise C
  - 10 EMG channels recorded from the forearm
  - 10 repetitions of each movement per subject

Pipeline overview:
  1. Load preprocessed EMG data from a pickle file
  2. Apply a Butterworth low-pass filter to smooth the signal
  3. Slice the signal into 200ms windows
  4. Extract time/frequency domain features from each window
  5. Split data into training and test sets by repetition index (2, 5, 7 for 
     testing)
  6. Normalize features and train a Random Forest classifier
  7. Evaluate accuracy of model
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
"""
Following paper specifications, db7 wavelet was used at level 3 even though 
it exceeds the recommended minimum signal length for a 20-sample window to 
maintain faithfulness to the original methodology. Resulting features are 
still valid and computable. Code suppresses warnings for this.
"""

import numpy as np
import pickle
import pywt
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


"""
=============================================================================
STEP 1: LOAD DATA
=============================================================================
The data is stored in a nested dictionary structure:
  emgData['ProcessedEMG'][subject][exercise][restimulus][rerepetition]

Each innermost value is a numpy array of shape (n_samples, n_sensors), where:
  - n_samples: number of time steps in that movement segment
  - n_sensors: number of EMG electrodes on the forearm (10 for DB1)

The 'restimulus' key tells us WHICH movement is being performed (label).
The 'rerepetition' key tells us WHICH repetition it is (used for train/test split).
=============================================================================
"""

with open('DB1processedEMG.pkl', 'rb') as f:
    emgData = pickle.load(f)


"""
=============================================================================
STEP 2: CONSTANTS
=============================================================================
DB1 was recorded at 100 Hz (100 samples per second).
A 200ms window therefore contains 20 samples for a 5 second rerepitiion
 (Actual sample count varies as rest period was disgarded).
We use 10 EMG channels and 20 histogram bins as specified in the paper.
=============================================================================
"""

FS          = 100                    
WINDOW_SIZE = int(0.200 * FS)        
N_CHANNELS  = 10                     
HIST_BINS   = 20                     


"""
=============================================================================
STEP 3: EXERCISE LABEL OFFSETS
=============================================================================
The 52 movements are split across 3 exercises, but each exercise resets its
restimulus labels back to 1. To give every movement a unique global label
(1-52), we add an offset based on which exercise we're in:

  Exercise 1 (E1): 12 finger movements       → labels 1-12  (offset 0)
  Exercise 2 (E2): 17 wrist/hand movements   → labels 13-29 (offset 12)
  Exercise 3 (E3): 23 grasping movements     → labels 30-52 (offset 29)
=============================================================================
"""

EXERCISE_OFFSETS = {
    'E1': 0,
    'E2': 12,
    'E3': 29
}


"""
=============================================================================
STEP 4: SIGNAL PROCESSING FUNCTIONS
=============================================================================
"""

def butter_lowpass_filter(data, cutoff=1.0, fs=FS, order=1):

    """
    Applies a 1st order Butterworth low-pass filter with a 1 Hz cutoff.

    The Otto Bock electrodes used in DB1 handle high-frequency noise, but
    this additional low-pass filter further smooths the signal by removing
    any remaining high-frequency components above 1 Hz before feature extraction.

    filtfilt applies the filter twice (forward + backward) to prevent any
    time shift in the signal.
    """

    nyq = 0.5 * fs                        
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)   


def get_windows(data):

    """
    Slices a continuous EMG segment into non-overlapping 200ms windows.

    We classify short windows of activity rather than individual samples 
    to reduce noise. Each window becomes one training example. 
    Any leftover samples at the end that don't fill a full window are discarded.

    Input:  (n_samples, 10) EMG array
    Output: list of (20, 10) arrays, one per window
    """

    windows = []
    for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
        windows.append(data[start:start + WINDOW_SIZE])
    return windows


"""
=============================================================================
STEP 5: FEATURE EXTRACTION FUNCTIONS
=============================================================================
Raw EMG values are not fed directly to the classifier. Instead, we extract
meaningful features from each 200ms window.
These features capture different aspects of the muscle activity pattern.

We compute 4 feature types across all 10 channels:
  - RMS:  signal energy / muscle activation level       →  10 values
  - TD:   shape and complexity of the signal            →  40 values
  - HIST: distribution of signal amplitudes             → 200 values
  - mDWT: frequency content via wavelet decomposition   →  40 values
                                              TOTAL:      290 values per window
=============================================================================
"""

def extract_rms(window):

    """
    Root Mean Square (RMS) — measures the average power/energy of the signal.
    Higher RMS means the muscle is contracting more strongly.
    One value per channel → 10 values total.
    """

    return np.sqrt(np.mean(window ** 2, axis=0))


def extract_td(window):

    """
    Hudgins Time Domain (TD) features — capture the shape of the EMG waveform.
    Four features per channel → 40 values total:

      MAV (Mean Absolute Value):  average signal amplitude
      WL  (Waveform Length):      total path length, captures signal complexity
      ZC  (Zero Crossings):       how often the signal crosses zero
      SSC (Slope Sign Changes):   how often the slope direction changes
    """

    mav   = np.mean(np.abs(window), axis=0)
    wl    = np.sum(np.abs(np.diff(window, axis=0)), axis=0)
    zc    = np.sum((window[:-1] * window[1:] < 0).astype(int), axis=0)
    diff1 = np.diff(window, axis=0)
    ssc   = np.sum((diff1[:-1] * diff1[1:] < 0).astype(int), axis=0)
    return np.concatenate([mav, wl, zc, ssc])


def extract_hist(window, n_bins=HIST_BINS):

    """
    Histogram (HIST) — captures the distribution of signal amplitudes.

    For each channel, the signal values are binned into 20 equally spaced
    intervals spanning ±3 standard deviations (the 3-sigma range covers
    ~99.7% of a normal distribution). The bin counts form the feature vector.
    This tells us how often the signal reaches different amplitude levels.
    20 bins × 10 channels → 200 values total.
    """

    features = []
    for ch in range(window.shape[1]):
        sig   = window[:, ch]
        sigma = np.std(sig)
        edges = np.linspace(-3 * sigma, 3 * sigma, n_bins + 1)
        hist, _ = np.histogram(sig, bins=edges)
        features.append(hist)
    return np.concatenate(features)


def extract_mdwt(window):

    """
    Marginal Discrete Wavelet Transform (mDWT) — captures frequency content.

    A db7 wavelet decomposes the signal into 3 frequency levels.
    The 'marginal' value for each sub-band is the sum of absolute coefficients,
    giving a single number per sub-band that represents energy at that frequency.
    4 coefficient arrays × 10 channels → 40 values total.
    """

    features = []
    for ch in range(window.shape[1]):
        sig      = window[:, ch]
        coeffs   = pywt.wavedec(sig, wavelet='db7', level=3)
        marginal = [np.sum(np.abs(c)) for c in coeffs]
        features.append(marginal)
    return np.array(features).flatten()


def extract_all_features(window):

    """
    Combines all four feature types into one flat feature vector.

    This combined representation gives the Random Forest the most complete
    picture of muscle activity. Normalization is applied later via 
    StandardScaler.

    Input:  (20, 10) window — 20 time samples across 10 EMG channels
    Output: (290,)   feature vector — ready to be fed to the classifier
    """

    rms  = extract_rms(window)
    td   = extract_td(window)
    hist = extract_hist(window)
    mdwt = extract_mdwt(window)
    return np.concatenate([rms, td, hist, mdwt])


"""
=============================================================================
STEP 6: BUILD FEATURE MATRIX (X) AND LABEL VECTOR (y)
=============================================================================
We iterate over every subject → exercise → movement → repetition,
filter and window the EMG signal, extract features from each window,
and accumulate them into three parallel arrays:

  X           — feature matrix, shape (total_windows, 290)
  y           — movement label for each window, shape (total_windows,)
  rep_indices — which repetition each window came from (used for train/test split)

All three arrays are always appended together so they stay perfectly aligned:
row i of X corresponds to label y[i] from repetition rep_indices[i].
=============================================================================
"""

X           = []
y           = []
rep_indices = []

for subject_key in tqdm(list(emgData['ProcessedEMG'].keys()), desc="Subjects", unit="subject"):
    for exercise_key in emgData['ProcessedEMG'][subject_key]:
        exercise = emgData['ProcessedEMG'][subject_key][exercise_key]

        exercise_id = exercise_key.split('_')[-1]
        offset      = EXERCISE_OFFSETS.get(exercise_id, 0)

        for restimulus_key in exercise:
            local_label = int(restimulus_key.split('_')[1])

            if local_label == 0:
                continue
            """
            Skip rest (label 0) — we only classify active movements
            in this implementation.
            """

            global_label = local_label + offset

            for rerepetition_key in exercise[restimulus_key]:
                rep_num = int(rerepetition_key.split('_')[1])

                emg = np.array(exercise[restimulus_key][rerepetition_key])

                if len(emg) < WINDOW_SIZE:
                    continue

                """
                Skip segments too short to form one 20-sample window.
                """

                emg_filtered = butter_lowpass_filter(emg)

                for window in get_windows(emg_filtered):
                    X.append(extract_all_features(window))
                    y.append(global_label)
                    rep_indices.append(rep_num)

print("\nConverting to numpy arrays...")
X           = np.array(X)
y           = np.array(y)
rep_indices = np.array(rep_indices)

print(f"X shape        : {X.shape}")
print(f"y shape        : {y.shape}")
print(f"Classes present: {np.unique(y)}")
print(f"Rep indices    : {np.unique(rep_indices)}")


"""
=============================================================================
STEP 7: TRAIN / TEST SPLIT
=============================================================================
Rather than splitting randomly, we split by repetition index.

Per the paper's methodology:
  Test set:     repetitions 2, 5, and 7  (~30% of data)
  Training set: all remaining repetitions (~70% of data)
=============================================================================
"""

test_reps  = [2, 5, 7]
train_mask = ~np.isin(rep_indices, test_reps)
test_mask  =  np.isin(rep_indices, test_reps)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")


"""
=============================================================================
STEP 8: FEATURE NORMALIZATION
=============================================================================
The four feature types (RMS, TD, HIST, mDWT) operate on very different scales.
Without normalization, larger-valued features would dominate the classifier.
StandardScaler transforms each feature to have mean=0 and std=1.
=============================================================================
"""

print("\nNormalizing features...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


"""
=============================================================================
STEP 9: TRAIN RANDOM FOREST CLASSIFIER
=============================================================================
A Random Forest is an ensemble of decision trees. Each tree votes on the
predicted movement class, and the majority vote wins. Key parameters:

  n_estimators=100:        number of trees in the forest
  class_weight='balanced': adjusts for any imbalance in class sizes
  n_jobs=-1:               use all available CPU cores for speed
  verbose=1:               print progress during training
=============================================================================
"""

print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf.fit(X_train, y_train)


"""
=============================================================================
STEP 10: EVALUATE
=============================================================================
We evaluate on the held-out test set (repetitions 2, 5, 7).
Key metrics:
  Accuracy:       overall % of correctly classified windows
  Chance level:   what random guessing would achieve (~1.92% for 52 classes)

The classification report shows per-class precision, recall and F1 score,
which helps identify which movements are hardest to classify.
=============================================================================
"""

y_pred   = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy       : {accuracy * 100:.2f}%")
print(f"Chance level   : {1/len(np.unique(y)) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))