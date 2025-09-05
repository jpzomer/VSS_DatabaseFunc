# This script converts the original CSV data files (raw experimental data) into a structured HDF5 dataset for vibration-based soft sensing experiments.
# It reads vibration and numerical measurement CSVs for each compressor unit and test, and stores them in a compressed, queryable HDF5 format.

import os
import re
import csv
import h5py
import warnings
import tqdm
import numpy as np
import pandas as pd

# Path to the folder containing the original CSV data
dataFolder = "D:/Rafael/Dados"

# List all files in the 'Dat' subfolder (numerical data)
allUnitsFolder = os.listdir(dataFolder+"/Dat")

# Extract unique compressor unit identifiers from filenames
allUnits = [re.findall("A.", unit) for unit in allUnitsFolder]  # Get all folder names with "Unidade"
allUnits = set([name[0][-1] for name in allUnits if len(name)>0])  # Filter for unique models

# Create a new HDF5 file to store the processed dataset
with h5py.File(f"{dataFolder}/dataset3.hdf5", "w") as fModel:
    # Iterate over each compressor unit
    for unitNum in tqdm.tqdm(allUnits,desc = "Compressor", position=0):
        # Create a group in the HDF5 file for this compressor unit
        unitGrp = fModel.create_group(unitNum)

        # Iterate over test types (e.g., 'A' = main map, 'B' = secondary map)
        for testType in tqdm.tqdm(['A','B'], desc = "   Mapa", leave=False, position=1):
            # Find all files for this unit and test type
            r = re.compile(f"{testType}{unitNum}.*")
            unitFiles = list(filter(r.match,allUnitsFolder))

            # For each test (experimental run)
            for testFile in tqdm.tqdm(unitFiles, desc = "      Teste", leave=False, position=2):
                testName = os.path.splitext(testFile)[0]
                # Create a group for this test
                testGrp = unitGrp.create_group(testName)

                # Parse test metadata from the filename
                testTags = testName.replace("[","").replace("]","").split("-")
                testGrp.attrs['type'] = testTags[0][0]  # Test type (A or B)
                testGrp.attrs['angularSpeed'] = testTags[1]  # Compressor speed
                testGrp.attrs['repetition'] = testTags[2]  # Repetition index
                testGrp.attrs['evaporatingTemperature'] = testTags[4]  # Evaporating temp
                testGrp.attrs['condensingTemperature'] = testTags[5]  # Condensing temp

                # --- Read and store the original CSV data ---
                # Vibration data (x, y, z axes)
                vibData = pd.read_table(f'{dataFolder}/Vib/{testFile}', delimiter = ';', decimal = '.', encoding='ANSI', header = None, names=['x', 'y', 'z'])
                # Numerical measurements (temperatures, pressures, etc.)
                numData = pd.read_table(f'{dataFolder}/Dat/{testFile}', delimiter = ';', decimal = '.', encoding='ANSI', header = None, names=['rpm', 't_evap_ref', 't_cond_ref','t_evap','t_cond','t_suc','t_comp','t_dis','p_suc','p_int','p_dis'])

                # Store vibration data in the HDF5 file
                vibMeas = testGrp.create_dataset("vibrationMeasurements", data = np.array(vibData), compression="gzip", shuffle=True)
                vibMeas.attrs['columnNames'] = list(vibData.columns)

                # Store numerical data in the HDF5 file
                numMeas = testGrp.create_dataset("numericalMeasurements", data = np.array(numData), compression="gzip", shuffle=True)
                numMeas.attrs['columnNames'] = list(numData.columns)