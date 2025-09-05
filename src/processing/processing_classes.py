
import numpy as np
import pandas as pd
import tqdm
import tensorflow as tf
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
import os

from .fft import raw_fft, ahryman_filter
# from .timefreq import ... (import as needed)


class BaseProcessing:
    """
    Flexible base class for dataset processing. Handles common logic for extracting metadata,
    looping over tests, and building the output DataFrame. Subclasses should implement process_slice,
    which returns a dict of features for each slice (keys = feature names, values = feature values).
    """
    def __init__(self, dataset_slice, dataset_list, opts):
        self.name = self.__class__.__name__
        self.dataset_slice = dataset_slice
        self.dataset_list = dataset_list
        self.opts = opts

    def extract_metadata(self, test):
        """Extracts and returns metadata for a test as a dict."""
        testTemperatures = np.float32(test.returnNumericalDataframe()[["t_evap", "t_cond"]].mean())
        testPressures = np.float32(test.returnNumericalDataframe()[["p_suc", "p_dis"]].mean())
        testConditions = test.returnAttributeDict()
        return {
            'unit': int(test.unit[1]),
            'rpm': int(testConditions["angularSpeed"]),
            't_evap_ref': -float(testConditions["evaporatingTemperature"].replace(',', '.')),
            't_cond_ref': float(testConditions["condensingTemperature"].replace(',', '.')),
            't_evap': testTemperatures[0],
            't_cond': testTemperatures[1],
            'p_suc': testPressures[0],
            'p_dis': testPressures[1],
        }

    def process(self):
        """
        Main processing loop. Subclasses should override process_slice to define
        how to process each set of vibration slices. process_slice must return a dict of features.
        """
        num_slices = self.opts['num_slices']
        rows = []
        for test in tqdm.tqdm(self.dataset_list, desc="Test", position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            metadata = self.extract_metadata(test)
            # By default, assume 3-axis vibration. Subclasses can override this logic if needed.
            testVibrationsX = test.splitVibrationWaveform(num_slices, "x")
            testVibrationsY = test.splitVibrationWaveform(num_slices, "y")
            testVibrationsZ = test.splitVibrationWaveform(num_slices, "z")
            for x, y, z in zip(testVibrationsX, testVibrationsY, testVibrationsZ):
                features = self.process_slice(x, y, z, test)
                row = {**metadata, **features}
                rows.append(row)
        return pd.DataFrame(rows)

    def process_slice(self, x, y, z, test):
        """
        Process a single set of vibration slices (x, y, z).
        Subclasses must override this method and return a dict of features.
        """
        raise NotImplementedError



class bandas_fft(BaseProcessing):
    """
    Processing class for FFT-based feature extraction using ahryman_filter.
    Returns frequency band features for x, y, z axes.
    """
    def __init__(self, dataset_slice, dataset_list, opts):
        super().__init__(dataset_slice, dataset_list, opts)
        self.name = 'ahryman_fft_10x_stdVib'

    def process_slice(self, x, y, z, test):
        num_slices = self.opts['num_slices']
        # Apply ahryman_filter to each axis
        fx = ahryman_filter(x, t=10/num_slices, dur=self.opts['dur'], sup=self.opts['sup'])
        fy = ahryman_filter(y, t=10/num_slices, dur=self.opts['dur'], sup=self.opts['sup'])
        fz = ahryman_filter(z, t=10/num_slices, dur=self.opts['dur'], sup=self.opts['sup'])
        return {'x': fx, 'y': fy, 'z': fz}


# Example: Time-domain statistics processing class
from scipy.stats import skew, kurtosis

class TimeStatsProcessing(BaseProcessing):
    """
    Example processing class for time-domain statistics (RMS, skewness, kurtosis) for each axis.
    Returns features: rms_x, rms_y, rms_z, skew_x, skew_y, skew_z, kurt_x, kurt_y, kurt_z
    """
    def process_slice(self, x, y, z, test):
        return {
            'rms_x': np.sqrt(np.mean(x**2)),
            'rms_y': np.sqrt(np.mean(y**2)),
            'rms_z': np.sqrt(np.mean(z**2)),
            'skew_x': skew(x),
            'skew_y': skew(y),
            'skew_z': skew(z),
            'kurt_x': kurtosis(x),
            'kurt_y': kurtosis(y),
            'kurt_z': kurtosis(z),
        }




# old version of bandas_fft for reference
# class bandas_fft_old:
#     def __init__(self, dataset_slice, dataset_list, opts):
#         self.name = 'ahryman_fft_10x_stdVib'
#         self.dataset_slice = dataset_slice
#         self.dataset_list = dataset_list
#         self.opts = opts
#     def process(self):
#         num_tests = len(self.dataset_list)
#         num_slices = self.opts['num_slices']
#         listunit = [None] * num_tests * num_slices
#         listrpm = [None] * num_tests * num_slices
#         listEvapRef = [None] * num_tests * num_slices
#         listCondRef = [None] * num_tests * num_slices
#         listX = [None] * num_tests * num_slices
#         listY = [None] * num_tests * num_slices
#         listZ = [None] * num_tests * num_slices
#         listEvap = [None] * num_tests * num_slices
#         listCond = [None] * num_tests * num_slices
#         listSuc = [None] * num_tests * num_slices
#         listDis = [None] * num_tests * num_slices
#         index = 0
#         for test in tqdm.tqdm(self.dataset_list,desc="Test",position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
#             testTemperatures = np.float32(test.returnNumericalDataframe()[["t_evap","t_cond"]].mean())
#             testPressures = np.float32(test.returnNumericalDataframe()[["p_suc","p_dis"]].mean())
#             testConditions = test.returnAttributeDict()
#             testVibrationsX = test.splitVibrationWaveform(num_slices, "x")
#             testVibrationsY = test.splitVibrationWaveform(num_slices, "y")
#             testVibrationsZ = test.splitVibrationWaveform(num_slices, "z")
#             for vib_slice in tqdm.tqdm(zip(testVibrationsX, testVibrationsY, testVibrationsZ), leave=False, desc = "     Slice", position=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
#                 listunit[index] = int(test.unit[1])
#                 listrpm[index] = (int(testConditions["angularSpeed"]))
#                 listEvapRef[index] = (-float(testConditions["evaporatingTemperature"].replace(',','.')))
#                 listCondRef[index] = (float(testConditions["condensingTemperature"].replace(',','.')))
#                 listX[index] = ahryman_filter(vib_slice[0],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
#                 listY[index] = ahryman_filter(vib_slice[1],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
#                 listZ[index] = ahryman_filter(vib_slice[2],t=10/num_slices,dur=self.opts['dur'],sup=self.opts['sup'])
#                 listEvap[index] = (testTemperatures[0])
#                 listCond[index] = (testTemperatures[1])
#                 listSuc[index] = (testPressures[0])
#                 listDis[index] = (testPressures[1])
#                 index = index+1
#         return pd.DataFrame({
#             'unit': listunit,
#             'rpm': listrpm,
#             't_evap_ref': listEvapRef,
#             't_cond_ref': listCondRef,
#             'x': listX,
#             'y': listY,
#             'z': listZ,
#             't_evap': listEvap,
#             't_cond': listCond,
#             'p_suc': listSuc,
#             'p_dis': listDis
#             })

# old version of raw_fft for reference, saved using parquet for multiple files that didn't fit in memory
# class raw_fft:
#     def __init__(self, dataset_slice, dataset_list, opts):
#         self.name = 'fft_10x_stdVib'
#         self.dataset_slice = dataset_slice
#         self.dataset_list = dataset_list
#         self.opts = opts
#         self.save_interval = opts.get('save_interval', 10)
#         self.df_temp = pd.DataFrame(columns=['unit', 'rpm', 't_evap_ref', 't_cond_ref', 'x', 'y', 'z', 't_evap', 't_cond'])

#     def process(self):
#         num_slices = self.opts['num_slices']
#         axes = ['x', 'y', 'z']
#         listunit, listrpm, listEvapRef, listCondRef = [], [], [], []
#         listX, listY, listZ, listEvap, listCond = [], [], [], [], []
#         output_dir = f'../processed_datasets/{self.name}/{self.dataset_slice}'
#         os.makedirs(output_dir, exist_ok=True)
#         def compute_fft_slices(vib_slices):
#             x, y, z = vib_slices
#             return raw_fft(x), raw_fft(y), raw_fft(z)
#         for i, test in enumerate(tqdm.tqdm(self.dataset_list, desc="Test", position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
#             testTemperatures = np.float32(test.returnNumericalDataframe()[["t_evap", "t_cond"]].mean())
#             testConditions = test.returnAttributeDict()
#             testVibrations = test.splitVibrationWaveforms(num_slices, axes)
#             evap_ref = -float(testConditions["evaporatingTemperature"].replace(',', '.'))
#             cond_ref = float(testConditions["condensingTemperature"].replace(',', '.'))
#             rpm = int(testConditions["angularSpeed"])
#             unit = int(test.unit[1])
#             with ThreadPoolExecutor() as executor:
#                 results = list(executor.map(compute_fft_slices, zip(testVibrations['x'], testVibrations['y'], testVibrations['z'])))
#             for fft_x, fft_y, fft_z in tqdm.tqdm(results, leave=False, desc="     Slice", position=1, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
#                 listunit.append(unit)
#                 listrpm.append(rpm)
#                 listEvapRef.append(evap_ref)
#                 listCondRef.append(cond_ref)
#                 listX.append(fft_x)
#                 listY.append(fft_y)
#                 listZ.append(fft_z)
#                 listEvap.append(testTemperatures[0])
#                 listCond.append(testTemperatures[1])
#             self.df_temp = pd.DataFrame({
#                 'unit': listunit,
#                 'rpm': listrpm,
#                 't_evap_ref': listEvapRef,
#                 't_cond_ref': listCondRef,
#                 'x': listX,
#                 'y': listY,
#                 'z': listZ,
#                 't_evap': listEvap,
#                 't_cond': listCond
#             })
#             if (i + 1) % self.save_interval == 0:
#                 self.df_temp.to_parquet(f'../processed_datasets/{self.name}/{self.dataset_slice}/{self.dataset_slice}_{i + 1}.parquet', compression='snappy')
#                 listunit.clear(); listrpm.clear(); listEvapRef.clear(); listCondRef.clear()
#                 listX.clear(); listY.clear(); listZ.clear(); listEvap.clear(); listCond.clear()
#                 self.df_temp = pd.DataFrame(columns=self.df_temp.columns)
#         final_df = pd.DataFrame({
#             'unit': listunit,
#             'rpm': listrpm,
#             't_evap_ref': listEvapRef,
#             't_cond_ref': listCondRef,
#             'x': listX,
#             'y': listY,
#             'z': listZ,
#             't_evap': listEvap,
#             't_cond': listCond
#         })
#         final_df.to_parquet(f'../processed_datasets/{self.name}/{self.dataset_slice}/{self.dataset_slice}_final.parquet', compression='snappy')
#         print("Final results saved.")
#         return final_df
