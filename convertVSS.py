import os
import re
import csv
import h5py
import warnings
import tqdm
import numpy as np
import pandas as pd


dataFolder = "D:/Rafael/Dados"

allUnitsFolder = os.listdir(dataFolder+"/Dat")

allUnits = [re.findall("A.", unit) for unit in allUnitsFolder] # Get all folder names with "Unidade"
allUnits = set([name[0][-1] for name in allUnits if len(name)>0]) # Filter for unique models

with h5py.File(f"{dataFolder}/dataset3.hdf5", "w") as fModel:
    for unitNum in tqdm.tqdm(allUnits,desc = "Compressor", position=0):
        
        unitGrp = fModel.create_group(unitNum) # Create new group for each compressor unit

        for testType in tqdm.tqdm(['A','B'], desc = "   Mapa", leave=False, position=1):

            r = re.compile(f"{testType}{unitNum}.*")
            unitFiles = list(filter(r.match,allUnitsFolder))


            for testFile in tqdm.tqdm(unitFiles, desc = "      Teste", leave=False, position=2):
                testName = os.path.splitext(testFile)[0]

                testGrp = unitGrp.create_group(testName)

                testTags = testName.replace("[","").replace("]","").split("-")
                testGrp.attrs['type'] = testTags[0][0]
                testGrp.attrs['angularSpeed'] = testTags[1]
                testGrp.attrs['repetition'] = testTags[2]
                testGrp.attrs['evaporatingTemperature'] = testTags[4]
                testGrp.attrs['condensingTemperature'] = testTags[5]


                # read test data
                vibData = pd.read_table(f'{dataFolder}/Vib/{testFile}', delimiter = ';', decimal = '.', encoding='ANSI', header = None, names=['x', 'y', 'z'])
                numData = pd.read_table(f'{dataFolder}/Dat/{testFile}', delimiter = ';', decimal = '.', encoding='ANSI', header = None, names=['rpm', 't_evap_ref', 't_cond_ref','t_evap','t_cond','t_suc','t_comp','t_dis','p_suc','p_int','p_dis'])

                vibMeas = testGrp.create_dataset("vibrationMeasurements", data = np.array(vibData), compression="gzip", shuffle=True)
                vibMeas.attrs['columnNames'] = list(vibData.columns)

                numMeas = testGrp.create_dataset("numericalMeasurements", data = np.array(numData), compression="gzip", shuffle=True)
                numMeas.attrs['columnNames'] = list(numData.columns)