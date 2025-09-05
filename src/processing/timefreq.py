import numpy as np
import librosa

def specgram_data(specgram):
    print('Image shape: {}\nFrequency resolution: {} Hz\nMax f: {} Hz\nf size: {}\nt size: {}'.\
    format(np.shape(specgram[0]), specgram[1][2]-specgram[1][1], specgram[1].max(), np.size(specgram[1]), np.size(specgram[2])))

def find_max_bins_cqt(fs,fmin=10,n_bins=200):
    fmax = fs
    while fmax > fs/2:
        n_bins -= 1
        fmax = librosa.cqt_frequencies(n_bins=n_bins,fmin=fmin).max()     
    return n_bins

def normalize_vibration(y):
    return (y-np.mean(y))/np.std(y)
