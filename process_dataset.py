
# =============================================================================
# process_dataset.py
#
# Script to process a selected slice of the vibration dataset using a specified
# feature extraction or processing class. Loads configuration, selects the
# processing method, applies it to the filtered dataset, and saves the output.
#
# Usage (from command line):
#   python process_dataset.py --method <ProcessingClass> --config <ConfigName> --slice <SliceName>
#
# Example:
#   python process_dataset.py --method bandas_fft --config fft_10x_stdVib --slice low
#
# Arguments:
#   --method   Name of the processing class to use (e.g., bandas_fft, raw_fft, etc.)
#   --config   Name of the processing configuration in configs/processing_configs.json
#   --slice    Name of the dataset slice to process (e.g., low, high, etc.)
#
# Output:
#   Saves a .pkl file with the processed DataFrame in processed_datasets/<method>/<slice>.pkl
# =============================================================================

import numpy as np
import pandas as pd
import json
import sys
import tensorflow as tf
import socket
import argparse
import importlib
import os

# Import project-specific utilities and database access
from src.core.dataset_utils import get_filter_attributes
from src.core.database import VSS_File


# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description='Process the selected dataset slice.')
parser.add_argument('--method', help='Processing method class to use')
parser.add_argument('--config', default='no_configs', help='Processing configuration to use')
parser.add_argument('--slice', type=str, default='low', help='The dataset slice to process.')
# parser.add_argument('--device', type=str, default='gpu:0', help='CUDA device to use.')
args = parser.parse_args()



# ----------------------
# Dynamically import the selected processing class
# ----------------------
def get_processing_class(class_name):
    """Dynamically import and return the processing class by name."""
    try:
        module = importlib.import_module('src.processing.processing_classes')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise Exception(f'Error loading processing class: {class_name}. Details: {e}')

if args.method:
    ProcessingClass = get_processing_class(args.method)
    print(f'Using class {args.method}')
else:
    raise Exception('No processing class provided.')



# ----------------------
# Load processing configuration parameters
# ----------------------
with open('./configs/processing_configs.json') as f:
    processing_config = json.load(f)

if args.config in processing_config:
    params = processing_config[args.config]
    print(f'Loaded params: {params}')
else:
    raise Exception(f'Unknown processing configuration: {args.config}')




# ----------------------
# Identify machine and set dataset path
# ----------------------
hostname = socket.gethostname()
print(f'Host: {hostname}')

with open('./configs/dataset_location.json') as f:
    config = json.load(f)
if hostname in config:
    datasetPath = config[hostname]
else:
    raise Exception(f'Unknown hostname: {hostname}')



# ----------------------
# Load dataset and process
# ----------------------
print(f'Loading dataset HDF5, slice: {args.slice}\n')
dataset = VSS_File(datasetPath)

# Filter the dataset according to the selected slice
filtered_list = dataset.DataframeAsList(get_filter_attributes(args.slice))

# Instantiate the processing class with the filtered data and options
processing_class = ProcessingClass(
    dataset_slice=args.slice,
    dataset_list=filtered_list,
    opts=params
)

# Run the processing and obtain the resulting DataFrame
df = processing_class.process()

# ----------------------
# Export the processed DataFrame to a pickle file
# ----------------------
output_path = f'./processed_datasets/{processing_class.name}/{args.slice}.pkl'
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
df.to_pickle(output_path)
print(f'Processed data saved to: {output_path}')