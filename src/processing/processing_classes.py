import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import os

from .fft import raw_fft, ahryman_filter
# from .timefreq import ... (import as needed)

class raw_fft:
    def __init__(self, dataset_slice, dataset_list, opts):
        self.name = 'fft_10x_stdVib'
        self.dataset_slice = dataset_slice
        self.dataset_list = dataset_list
        self.opts = opts
        self.save_interval = opts.get('save_interval', 10)
        self.df_temp = pd.DataFrame(columns=['unit', 'rpm', 't_evap_ref', 't_cond_ref', 'x', 'y', 'z', 't_evap', 't_cond'])

    def process(self):
        num_slices = self.opts['num_slices']
        axes = ['x', 'y', 'z']
        listunit, listrpm, listEvapRef, listCondRef = [], [], [], []
        listX, listY, listZ, listEvap, listCond = [], [], [], [], []
        output_dir = f'../processed_datasets/{self.name}/{self.dataset_slice}'
        os.makedirs(output_dir, exist_ok=True)
        def compute_fft_slices(vib_slices):
            x, y, z = vib_slices
            return raw_fft(x), raw_fft(y), raw_fft(z)
        for i, test in enumerate(tqdm.tqdm(self.dataset_list, desc="Test", position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            testTemperatures = np.float32(test.returnNumericalDataframe()[["t_evap", "t_cond"]].mean())
            testConditions = test.returnAttributeDict()
            testVibrations = test.splitVibrationWaveforms(num_slices, axes)
            evap_ref = -float(testConditions["evaporatingTemperature"].replace(',', '.'))
            cond_ref = float(testConditions["condensingTemperature"].replace(',', '.'))
            rpm = int(testConditions["angularSpeed"])
            unit = int(test.unit[1])
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(compute_fft_slices, zip(testVibrations['x'], testVibrations['y'], testVibrations['z'])))
            for fft_x, fft_y, fft_z in tqdm.tqdm(results, leave=False, desc="     Slice", position=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                listunit.append(unit)
                listrpm.append(rpm)
                listEvapRef.append(evap_ref)
                listCondRef.append(cond_ref)
                listX.append(fft_x)
                listY.append(fft_y)
                listZ.append(fft_z)
                listEvap.append(testTemperatures[0])
                listCond.append(testTemperatures[1])
            self.df_temp = pd.DataFrame({
                'unit': listunit,
                'rpm': listrpm,
                't_evap_ref': listEvapRef,
                't_cond_ref': listCondRef,
                'x': listX,
                'y': listY,
                'z': listZ,
                't_evap': listEvap,
                't_cond': listCond
            })
            if (i + 1) % self.save_interval == 0:
                self.df_temp.to_parquet(f'../processed_datasets/{self.name}/{self.dataset_slice}/{self.dataset_slice}_{i + 1}.parquet', compression='snappy')
                listunit.clear(); listrpm.clear(); listEvapRef.clear(); listCondRef.clear()
                listX.clear(); listY.clear(); listZ.clear(); listEvap.clear(); listCond.clear()
                self.df_temp = pd.DataFrame(columns=self.df_temp.columns)
        final_df = pd.DataFrame({
            'unit': listunit,
            'rpm': listrpm,
            't_evap_ref': listEvapRef,
            't_cond_ref': listCondRef,
            'x': listX,
            'y': listY,
            'z': listZ,
            't_evap': listEvap,
            't_cond': listCond
        })
        final_df.to_parquet(f'../processed_datasets/{self.name}/{self.dataset_slice}/{self.dataset_slice}_final.parquet', compression='snappy')
        print("Final results saved.")
        return final_df

class bandas_fft:
    def __init__(self, dataset_slice, dataset_list, opts):
        self.name = 'ahryman_fft_10x_stdVib'
        self.dataset_slice = dataset_slice
        self.dataset_list = dataset_list
        self.opts = opts
    def process(self):
        num_tests = len(self.dataset_list)
        num_slices = self.opts['num_slices']
        listunit = [None] * num_tests * num_slices
        listrpm = [None] * num_tests * num_slices
        listEvapRef = [None] * num_tests * num_slices
        listCondRef = [None] * num_tests * num_slices
        listX = [None] * num_tests * num_slices
        listY = [None] * num_tests * num_slices
        listZ = [None] * num_tests * num_slices
        listEvap = [None] * num_tests * num_slices
        listCond = [None] * num_tests * num_slices
        listSuc = [None] * num_tests * num_slices
        listDis = [None] * num_tests * num_slices
        index = 0
        for test in tqdm.tqdm(self.dataset_list,desc="Test",position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            testTemperatures = np.float32(test.returnNumericalDataframe()[["t_evap","t_cond"]].mean())
            testPressures = np.float32(test.returnNumericalDataframe()[["p_suc","p_dis"]].mean())
            testConditions = test.returnAttributeDict()
            testVibrationsX = test.splitVibrationWaveform(num_slices, "x")
            testVibrationsY = test.splitVibrationWaveform(num_slices, "y")
            testVibrationsZ = test.splitVibrationWaveform(num_slices, "z")
            for vib_slice in tqdm.tqdm(zip(testVibrationsX, testVibrationsY, testVibrationsZ), leave=False, desc = "     Slice", position=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                listunit[index] = int(test.unit[1])
                listrpm[index] = (int(testConditions["angularSpeed"]))
                listEvapRef[index] = (-float(testConditions["evaporatingTemperature"].replace(',','.')))
                listCondRef[index] = (float(testConditions["condensingTemperature"].replace(',','.')))
                listX[index] = ahryman_filter(vib_slice[0],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
                listY[index] = ahryman_filter(vib_slice[1],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
                listZ[index] = ahryman_filter(vib_slice[2],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
                listEvap[index] = (testTemperatures[0])
                listCond[index] = (testTemperatures[1])
                listSuc[index] = (testPressures[0])
                listDis[index] = (testPressures[1])
                index = index+1
        return pd.DataFrame({
            'unit': listunit,
            'rpm': listrpm,
            't_evap_ref': listEvapRef,
            't_cond_ref': listCondRef,
            'x': listX,
            'y': listY,
            'z': listZ,
            't_evap': listEvap,
            't_cond': listCond,
            'p_suc': listSuc,
            'p_dis': listDis
            })
