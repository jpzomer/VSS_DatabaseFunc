import json
import numpy as np
import os

def get_filter_attributes(dataset_slice):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    slices_path = os.path.join(script_dir, '..', '..', 'configs', 'slices.json')
    with open(slices_path, 'r') as f:
        slices = json.load(f)
    return slices.get(dataset_slice, None)

def create_filter_dict(angularSpeed, compressor):
    result = {}
    if angularSpeed is not None:
        result["angularSpeed"] = angularSpeed
    if compressor is not None:
        result["compressor"] = compressor
    return result

def get_total_memory(df):
    return df.memory_usage(index=True, deep=True).sum()/(1024**3)

# def amplitude_to_db(amostra_fft,db_ref = 5*(10**-8)):
#     return 20*np.log10(amostra_fft/db_ref)

# def df_max(df):
#     return max([df[col].apply(np.max).max() for col in ['x','y','z']])

# def df_min(df):
#     return min([df[col].apply(np.min).min() for col in ['x','y','z']])
