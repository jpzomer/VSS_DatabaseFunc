import h5py
import pandas as pd
import numpy as np
from contextlib import contextmanager

@contextmanager
def open_hdf5_file(filePath):
    file = h5py.File(filePath, 'r')
    try:
        yield file
    finally:
        file.close()

class VSS_File:
    """Class for vibration-based soft sensing database in an hdf5 file"""
    def __init__(self, filePath):
        self.path = filePath
        self._index = 0
        self._fileh5ref = h5py.File(filePath,'r')
        self.units =  [self.VSS_Unit_Reference(self,self._fileh5ref[group]) for group in self._fileh5ref.keys()]

    def __repr__(self):
        return f"Vibration-based database for ({len(self.units)} units)"

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index == len(self.units):
            self._index = 0
            raise StopIteration
        else:
            self._index += 1
            return self.units[self._index-1]
    
    def DataframeAsList(self, attributeDict, selectedUnits = None):
        if selectedUnits is None:
            selectedUnits = self.units
        else:
            selectedUnits = [self.VSS_Unit_Reference(self,self._fileh5ref[group]) for group in selectedUnits]
        return [item for unit in selectedUnits for item in unit.filterTestsByAttributeDict(attributeDict)]

    class VSS_Unit_Reference:
        def __init__(self, parent, unitGroupId:h5py.Group):
            self._h5ref = unitGroupId
            self._h5file = parent
            self.name = unitGroupId.name
            self._index = 0
            self.tests = [self.VSS_Test_Reference(self,self._h5ref[group]) for group in self._h5ref.keys()]

        def __repr__(self):
            return f"Vibration-based database for unit <{self.name}> ({len(self.tests)} tests)"
        def __str__(self):
            return self.name
        def __iter__(self):
            return self
        def __next__(self):
            if self._index == len(self.tests):
                self._index = 0
                raise StopIteration
            else:
                self._index += 1
                return self.tests[self._index-1]
        def filterTestsByAttributeDict(self, attributeDict):
            output = [test for test in self.tests]
            if "angularSpeed" in attributeDict.keys():
                output = [test for test in output if int(test._h5ref.attrs['angularSpeed']) in attributeDict['angularSpeed']]
            if "condensingTemperature" in attributeDict.keys():
                output = [test for test in output if min(attributeDict['condensingTemperature']) <= float(test._h5ref.attrs['condensingTemperature'].replace(',','.')) <= max(attributeDict['condensingTemperature'])]
            if "evaporatingTemperature" in attributeDict.keys():
                output = [test for test in output if min(attributeDict['evaporatingTemperature']) <= float(test._h5ref.attrs['evaporatingTemperature'].replace(',','.')) <= max(attributeDict['evaporatingTemperature'])]
            if "compressor" in attributeDict.keys():
                output = [test for test in output if int(str(test.name)[1]) in attributeDict['compressor']]
            return output

        class VSS_Test_Reference:
            def __init__(self, parent, testGroupId:h5py.Group):
                self._h5ref = testGroupId
                self._h5file = parent._h5file
                self.h5unit = parent
                self.date = testGroupId.name
                self.unit = parent.name
                self.name = testGroupId.name
            def __repr__(self):
                return f"Vibration soft sensing test database <{self.name}>"
            def __str__(self):
                return (self.name)
            def returnNumericalDatabase(self):
                return np.array(self._h5ref["numericalMeasurements"])
            def returnNumericalHeaders(self):
                return list(self._h5ref["numericalMeasurements"].attrs["columnNames"])
            def returnNumericalDataframe(self):
                return pd.DataFrame(data = self.returnNumericalDatabase(), columns = self.returnNumericalHeaders())
            def returnVibrationDatabase(self):
                return np.array(self._h5ref["vibrationMeasurements"])
            def returnVibrationHeaders(self):
                return list(self._h5ref["vibrationMeasurements"].attrs["columnNames"])
            def returnVibrationDataframe(self):
                return pd.DataFrame(data = self.returnVibrationDatabase(), columns = self.returnVibrationHeaders())
            def returnAttributeList(self):
                return list(self._h5ref.attrs)
            def returnAttributeDict(self):
                AttributeList = self.returnAttributeList()
                AttributeDict = {}
                for element in AttributeList:
                    AttributeDict[element] = self._h5ref.attrs[element]
                return AttributeDict
            def splitVibrationWaveform(self, n, axis):
                vibData = np.array(self.returnVibrationDataframe()[axis])
                if np.size(vibData) % n:
                    return np.split(vibData[0:-(np.size(vibData)%n)],n)
                else:
                    return np.split(vibData,n)
            def splitVibrationWaveforms(self, n, axes):
                for axis in axes:
                    if axis not in self.returnVibrationHeaders():
                        raise ValueError(f"Invalid axis: {axis}")
                vibDataList = [self.returnVibrationDataframe()[axis].to_numpy() for axis in axes]
                results = {}
                for i, vibData in enumerate(vibDataList):
                    size = vibData.size
                    if size % n:
                        vibData = vibData[0:-(size % n)]
                    results[axes[i]] = vibData.reshape(n, -1)
                return results
