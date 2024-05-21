import h5py
import pandas as pd
import numpy as np

class VSS_File:
    # Class for vibration-based soft sensing database in an hdf5 file
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
        
    # adaptar pra escolher apenas alguns compressores
    def DataframeAsList(self, attributeDict, selectedUnits = None):
        if selectedUnits is None:
            selectedUnits = self.units
        else:
            selectedUnits = [self.VSS_Unit_Reference(self,self._fileh5ref[group]) for group in selectedUnits]

        return [item for sublist in \
                [unit.filterTestsByAttributeDict(attributeDict) for unit in selectedUnits] \
                      for item in sublist]
    

    # def returnDataframe(self, attributeDict):
    #     dataList = self.DataframeAsList(attributeDict)
    #     df = pd.DataFrame(np.mean(np.array(dataset.DataframeAsList(attributeDict)[0]._h5ref["numericalMeasurements"]), axis=0))
    #     return df


    class VSS_Unit_Reference:
        # Class for a unit group inside a hdf5 file
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
            
        # def filterTestsByAttribute(self, attribute, value):
        #     match attribute:
        #         case "angularSpeed":
        #             # 2100 2475 2850 3225 3600 RPM
        #             return [test for test in self.tests if test._h5ref.attrs['angularSpeed'] in value]
        #         case "condensingTemperature" | "evaporatingTemperature":
        #             # condensingTemperature 34º C até 54 ºC
        #             # evaporatingTemperature 10º C até 30 ºC
        #             return [test for test in self.tests if min(value) <= test._h5ref.attrs[attribute] <= max(value)]
        #         # case "repetition":
        #         #     #  n
        #         #     pass
        #         case "type":
        #             # A = mapa principal
        #             # B = mapa secundário
        #             return [test for test in self.tests if test._h5ref.attrs['type'] == value]
                
        def filterTestsByAttributeDict(self, attributeDict):
                output = [test for test in self.tests]
                if "angularSpeed" in attributeDict.keys():
                    output = [test for test in output if int(test._h5ref.attrs['angularSpeed']) \
                               in attributeDict['angularSpeed']]
                else:
                    pass

                if "condensingTemperature" in attributeDict.keys():
                    output = [test for test in output if \
                             min(attributeDict['condensingTemperature']) \
                                <= float(test._h5ref.attrs['condensingTemperature'].replace(',','.')) \
                                    <= max(attributeDict['condensingTemperature'])]
                else:
                    pass

                if "evaporatingTemperature" in attributeDict.keys():
                    output = [test for test in output if \
                             min(attributeDict['evaporatingTemperature']) \
                                <= float(test._h5ref.attrs['evaporatingTemperature'].replace(',','.')) \
                                    <= max(attributeDict['evaporatingTemperature'])]
                else:
                    pass

                if "repetition" in attributeDict.keys():
                    pass
                else:
                    pass

                if "type" in attributeDict.keys():
                    pass
                else:
                    pass
                
                #adição de "compressor"
                if "compressor" in attributeDict.keys():
                    output = [test for test in output if int(str(test.name)[1]) in attributeDict['compressor']]
                return output


        class VSS_Test_Reference:
            # Class for a test group inside a hdf5 file
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
        
#if __name__ == "__main__":
#