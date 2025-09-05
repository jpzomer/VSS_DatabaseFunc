# VSS_DatabaseFunc

## Overview
This repository provides tools and utilities for managing, processing, and extracting features from vibration-based soft-sensing datasets for compressor units. It supports HDF5 storage, advanced signal processing, and feature engineering for machine learning and analysis workflows.

## Project Structure
- `src/core/`: HDF5 database access, dataset utilities, and filtering.
- `src/processing/`: Signal processing and feature extraction classes (FFT, time-domain, etc.).
- `src/utils/`: Utility scripts (e.g., train/val split generation).
- `configs/`: JSON config files for dataset locations, processing, and data splits.
- `processed_datasets/`: Output directory for processed feature datasets.
- `convertVSS.py`: Converts raw CSVs to HDF5 format.
- `process_dataset.py`: Main script for processing and feature extraction.
- `onboarding_guide.ipynb`: Step-by-step notebook for onboarding and data exploration.

## Getting Started
1. **Environment Setup**
	- Create the conda environment:
	  ```bash
	  conda env create -f environment.yml
	  conda activate vss_dataset
	  ```
2. **Configure Dataset Location**
	- Edit `configs/dataset_location.json` to set the HDF5 path for your machine.
3. **Convert Raw Data (if needed)**
	- Run `convertVSS.py` to convert CSVs to HDF5.
4. **Process Dataset**
	- Example command:
	  ```bash
	  python process_dataset.py --method bandas_fft --config fft_10x_stdVib --slice low
	  ```
	- Output will be saved in `processed_datasets/<method>/<slice>.pkl`.
5. **Explore Data**
	- Use `onboarding_guide.ipynb` for interactive exploration and visualization.

## Feature Extraction
- Add new processing classes in `src/processing/processing_classes.py`.
- Each class should inherit from `BaseProcessing` and implement `process_slice` to return a dict of features per slice.
- See `bandas_fft` and `TimeStatsProcessing` for examples.

## Example: Loading a Processed Dataset
```python
import pandas as pd
df = pd.read_pickle(r"processed_datasets/ahryman_fft_10x_stdVib/low.pkl")
# or, for parquet:
# df = pd.read_parquet("processed_datasets/ahryman_fft_10x_stdVib/low.parquet")
```

## Contributing
- Please open issues or pull requests for bug fixes, improvements, or new features.
- Follow the code style and add docstrings/comments for clarity.

## License
MIT License (see LICENSE file).
