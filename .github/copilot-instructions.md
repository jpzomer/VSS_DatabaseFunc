# Copilot Instructions for VSS_DatabaseFunc

## Project Overview
- This repository manages vibration-based soft-sensing datasets for compressor units, focusing on HDF5 storage, processing, and feature extraction.
- Main components:
  - `src/core/database.py`: HDF5 database access and iteration (see `VSS_File` and nested reference classes).
  - `src/processing/`: Signal processing and feature extraction (notably `processing_classes.py` with `bandas_fft`, `bandas_fft_new`, `raw_fft`).
  - `src/utils/`: Utility scripts (e.g., `generate_indices.py` for train/val splits).
  - `convertVSS.py`: Converts raw CSVs to HDF5, parses metadata, and stores vibration/test data.
  - `configs/`: JSON config files for dataset locations, processing, and data splits.

## Key Patterns & Conventions
- **Data Model:**
  - HDF5 structure: Top-level groups = compressor units; subgroups = tests; attributes store metadata (see `convertVSS.py`).
  - Iteration: `VSS_File` exposes `.units`, each unit has `.tests`, all are iterable (see `test.ipynb`).
- **Processing:**
  - Feature extraction classes (`bandas_fft`, `bandas_fft_new`, `raw_fft`) expect a dataset list and config options (`opts`).
  - Use `process()` to generate DataFrames with features; see column conventions in `processing_classes.py`.
- **Config-driven:**
  - Processing and slicing parameters are loaded from JSON in `configs/`.
  - Dataset location is machine-dependent (see `dataset_location.json`).
- **Reproducibility:**
  - Train/val splits are generated with fixed seeds (`generate_indices.py`).

## Developer Workflows
- **Environment:**
  - Use `environment.yml` to create the `vss_dataset` conda environment.
- **Testing/Debugging:**
  - Use `test.ipynb` for interactive exploration and validation of data/model logic.
  - Launch/debug Python scripts via VS Code (see `.vscode/launch.json`).
- **Adding Processing Steps:**
  - Add new classes to `src/processing/processing_classes.py` following the `process()` pattern.
  - Reference config parameters via `self.opts`.

## Integration & Dependencies
- Core dependencies: `h5py`, `numpy`, `pandas`, `tqdm` (see `environment.yml`).
- Data flows: CSV → HDF5 (via `convertVSS.py`) → DataFrame (via processing classes).

## Examples
- See `test.ipynb` for usage patterns: loading datasets, iterating units/tests, extracting DataFrames.
- Example: `dataset.units[0].tests[0].returnNumericalDataframe()` returns a DataFrame for a test.

## Special Notes
- Always set `HDF5_USE_FILE_LOCKING=FALSE` before importing `h5py` if running on Windows with file sharing.
- All submodules in `src/` are Python packages (see `__init__.py` files).
