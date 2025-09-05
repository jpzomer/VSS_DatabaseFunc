import numpy as np
import librosa

def normalize_vibration(y):
    return (y-np.mean(y))/np.std(y)