import numpy as np
import pandas as pd
import json
import sys
import tensorflow as tf
import socket
import argparse
import importlib
import os


# Updated imports for new src structure
from src.core.dataset_utils import get_filter_attributes
from src.core.database import VSS_File

# Create the parser
parser = argparse.ArgumentParser(description='Process the selected dataset slice.')
# Add an argument for the dataset slice
parser.add_argument('--method', help='Processing method class to use')
parser.add_argument('--config', default='no_configs', help='Processing configuration to use')
parser.add_argument('--slice', type=str, default='low', help='The dataset slice to process.')
# parser.add_argument('--device', type=str, default='gpu:0', help='CUDA device to use.')
# Parse the arguments
args = parser.parse_args()


# Updated to import from new processing folder
def get_processing_class(class_name):
    try:
        module = importlib.import_module('src.processing.processing_classes')
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise Exception(f'Error loading processing class: {class_name}. Details: {e}')

# Use the processing class
if args.method:
    ProcessingClass = get_processing_class(args.method)
    print(f'Using class {args.method}')
else:
    raise Exception('No processing class provided.')


# CONFIGURAÇÕES DA CLASSE DE PROCESSAMENTO (opts)
with open('./configs/processing_configs.json') as f:
    processing_config = json.load(f)

# Extract the parameters
if args.config in processing_config:
    params = processing_config[args.config]
    print(f'loaded params {params}')
else:
    raise Exception(f'Unknown processing configuration: {args.config}')



# IDENTIFICAÇÃO DA MÁQUINA RODANDO O SCRIPT
hostname = socket.gethostname()
print(f'host: {hostname}')

# configuração do caminho do dataset hdf5
with open('./configs/dataset_location.json') as f:
    config = json.load(f)
# Set the datasetPath based on the hostname
if hostname in config:
    datasetPath = config[hostname]
else:
    raise Exception(f'Unknown hostname: {hostname}')


# CARREGAMENTO DE DADOS
print(f'carregando dataset hdf5, slice: {args.slice}\n')
dataset = VSS_File(datasetPath)


# definindo os parâmetros de filtragem abaixo
processing_class = ProcessingClass(dataset_slice=args.slice,
                           dataset_list=dataset.DataframeAsList(get_filter_attributes(args.slice)),
                           opts = params)


df = processing_class.process()


# e exporto
output_path = f'./processed_datasets/{processing_class.name}/{args.slice}.pkl'
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
df.to_pickle(output_path)