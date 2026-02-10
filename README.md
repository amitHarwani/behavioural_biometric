# Multi-User Multi-Modal Continuous Authentication using Behavioural Biometrics 

This repository contains preprocessing, training, evaluation and experiments for behavioual-biometric authentication using the HMOG public dataset. It includes preprocessing utilities, multiple experiment variants (exp1..exp5), model definitions, training scripts, evaluation and result notes.

**Project Structure**
- `preprocess.py`: full preprocessing pipeline that synchronizes modalities (keystroke, accelerometer, gyroscope, magnetometer, touch), extracts features, bins by time window and writes per-split pickles.
- `RUN_NOTES.md`: experiment summaries and results (EERs, training strategies, checkpoints used).
- `exp1/` .. `exp5/`: experiment folders. Each contains `data_loader.py`, `model_*.py`, `train_*.py`, `evaluate.py`, and `validation.py` for that experiment.

**Dataset**
- The preprocessing expects the HMOG dataset under `./datasets/hmog/public_dataset` (see `preprocess.py` constant `HMOG_DATASET_PATH`).
- You must download and place the HMOG public dataset in that path before running preprocessing.

Overview of data flow
- `preprocess.py` reads modality CSVs per user/session and:
  - Cleans keystroke and sensor/touch data (filters orientation, invalid timestamps, keycode ranges).
  - Extracts features: keystroke digrams/trigrams, hold latency; IMU/touch derivatives (first/second-order gradients).
  - Synchronizes modalities either by binning (`sync_by_binning`).
  - Splits users into train/validation/test sets and writes pickles named like `v1_training_users_data_tw{TIME_WINDOW}ms.pickle`.

Key preprocessing notes
- `TIME_WINDOW` controls the window size used to bin/synchronize modalities (in ms).
- `preprocess.py` removes user `733162` by default due to missing accelerometer data (see script).

Quick start
1. Ensure dataset is available at `./datasets/hmog/public_dataset`.
2. Install dependencies (see Dependencies section).
3. Run preprocessing (adjust `TIME_WINDOW` or random seeds in script if needed):

```powershell
python preprocess.py
```

4. Train a model (example using exp3_2):

```powershell
cd exp3_2
python train_3_2.py
```

5. Evaluate/validate using the scripts in the experiment folder, e.g.: `evaluate.py` or `validation.py`.


Results & notes
- See `RUN_NOTES.md` for experiment-by-experiment EERs and training hyperparameters. The best reported Mahalanobis EER in these notes is from `exp3_2` (epoch 39) with value ~1.91% (see `RUN_NOTES.md`).

Dependencies
- Python 3.8+ recommended.
- Key Python packages required (install via `pip`):
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `torch` (PyTorch)
  - `matplotlib` (for plots/visualizations)


