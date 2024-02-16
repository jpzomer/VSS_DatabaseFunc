import h5py
import pandas as pd

class VSS_File:
    # Class for vibration-based soft sensing database in an hdf5 file
    def __init__(self, filePath):
        self.path = filePath
        self._index = 0
        self._fileh5ref = h5py.File(filePath,'r')
        # self._modelh5ref = [self._fileh5ref[key] for key in self._fileh5ref.keys()][0]
        self.units =  [self.VSS_Unit_Reference(self,self._fileh5ref[group]) for group in self._fileh5ref.keys()]

    # def __repr__(self):
    #     return f"Vibration-based database <{self.model}> ({len(self.units)} units)"
    
    # def __str__(self):
    #     return self.model
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index == len(self.units):
            self._index = 0
            raise StopIteration
        else:
            self._index += 1
            return self.units[self._index-1]

    class VSS_Unit_Reference:
        # Class for a unit group inside a hdf5 file
        def __init__(self, parent, unitGroupId:h5py.Group):
            self._h5ref = unitGroupId
            self._h5file = parent
            self.name = unitGroupId.name
            self._index = 0
            self.tests = [self.VSS_Test_Reference(self,self._h5ref[group]) for group in self._h5ref.keys()]

        def __repr__(self):
            return f"Run-in unit database <{self.name}> ({len(self.tests)} tests)"
    
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

        class VSS_Test_Reference:
            # Class for a test group inside a hdf5 file
            def __init__(self, parent, testGroupId:h5py.Group):
                self._h5ref = testGroupId
                self._h5file = parent._h5file
                self.h5unit = parent
                self.date = testGroupId.name
                self.unit = parent.name

            def __repr__(self):
                return f"Run-in test database <{self.name}>"
    
            def __str__(self):
                return (self.unit + "/" + self.date)

            def getMeasurementsDataframe(self, varName: list[str], tStart:float = None, tEnd:float = None, indexes: list[int] = None):
                # Returns a dataframe containing the measurements of the desired indexes or time range

                if (indexes is not None) and ((tEnd is not None) or (tStart is not None)):
                    raise Exception("Both index and time range provided. Only one allowed.")
                
                allVars = self.getVarNames()

                # Check vars
                for var in varName:
                    if var not in allVars:
                        raise Exception("One or more variables are not available for the selected test. Run getVarNames() to list all available variables.")  

                
                if indexes is not None: # Get by index
                    df = pd.DataFrame(index = indexes,columns = ["time"] + varName)

                    for ind in indexes:
                        df.at[ind, "time"] = self._h5ref[str(ind)].attrs["time"]

                        for var in varName:
                            if var in ["voltageRAW","acousticEmissionRAW", "currentRAW",
                                    "vibrationRAWLongitudinal", "vibrationRAWRig", "vibrationRAWLateral"]:
                                # Get values from high frequency dataset
                                df.at[ind, var] = self._h5ref[str(ind)][var][()]
                            else:
                                df.at[ind, var] = self._h5ref[str(ind)].attrs[var]
                    
                    return df
                
                else: # Get by time
                    if tStart > tEnd:
                        raise Exception("Starting time should be smaller than end time.")

                    allTests = list(self._h5ref.keys())
                    df = pd.DataFrame(columns = ["time"] + varName)

                    count = 0
                    t_act = self._h5ref["0"].attrs["time"]

                    while t_act <= tEnd:

                        if t_act > tStart:
                            df.loc[len(df)] = pd.Series(dtype='float64') # Initializes new row
                            df.index[len(df)-1] = count
                            df.at[count, "time"] = t_act

                            for var in varName:
                                if var in ["voltageRAW","acousticEmissionRAW", "currentRAW",
                                        "vibrationRAWLongitudinal", "vibrationRAWRig", "vibrationRAWLateral"]:
                                    # Get values from high frequency dataset
                                    df.at[count, var] = self._h5ref[str(count)][var][()]
                                else:
                                    df.at[count, var] = self._h5ref[str(count)].attrs[var]

                        count = count + 1

                        if str(count) not in allTests: # All measurements have been checked
                            break
                        else:
                            t_act = self._h5ref[str(count)].attrs["time"]


            def getVarNames(self):
                # Return the name of all measurement variables of a given test

                # List of all available variables based on the 1st measurement of the test
                return list(self._h5ref["0"].attrs.keys())+list(self._h5ref["0"].keys())
        
#if __name__ == "__main__":
#