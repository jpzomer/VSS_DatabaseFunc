import numpy as np

def raw_fft(y, Fs=51200):
    size = len(y)
    ft = np.fft.fft(y)
    P2 = np.abs(ft / size)
    ft = P2[:size // 2 + 1]
    ft[1:-1] = 2 * ft[1:-1]
    return ft

def band_filter(ft,t=1,dur=200,sup=0.1,fim=25600,db_ref = 5*(10**-8)):
    sup = dur * sup
    freq = 1 / t  # step frequency of each FFT point
    fim = 25600
    dur_points = round(dur / freq)  # duration of the range in points
    sup_points = round(sup / freq)  # duration of the overlap in points
    len_fft = round(fim / freq)  # total number of points
    n_bands_fft = int(1 + np.ceil((len_fft - dur_points) / (dur_points - sup_points)))  # number of FFT bands
    i_start = 0
    amostra_fft = np.empty(n_bands_fft)
    for j in range(n_bands_fft - 1):
        band = ft[i_start:i_start + dur_points]
        amostra_fft[j] = np.sum(band ** 2)  # energy
        i_start = i_start + dur_points - sup_points
    band = ft[i_start:len_fft]  # last band
    amostra_fft[n_bands_fft - 1] = np.sum(band ** 2)  # energy
    return 20*np.log10(amostra_fft/db_ref)

def ahryman_filter(y,Fs=51200,t=1,dur=200,sup=0.1,db_ref = 5*(10**-8)):
    size = int(Fs * t)
    sup = dur * sup
    freq = 1 / t  # step frequency of each FFT point
    fim = 25600
    dur_points = round(dur / freq)  # duration of the range in points
    sup_points = round(sup / freq)  # duration of the overlap in points
    len_fft = round(fim / freq)  # total number of points
    n_bands_fft = int(1 + np.ceil((len_fft - dur_points) / (dur_points - sup_points)))  # number of FFT bands
    ft = np.fft.fft(y)  # calculates FFT of v
    P2 = np.abs(ft / size)
    ft = P2[:size // 2 + 1]
    ft[1:-1] = 2 * ft[1:-1]
    i_start = 0
    amostra_fft = np.empty(n_bands_fft)
    for j in range(n_bands_fft - 1):
        band = ft[i_start:i_start + dur_points]
        amostra_fft[j] = np.sum(band ** 2)  # energy
        i_start = i_start + dur_points - sup_points
    band = ft[i_start:len_fft]  # last band
    amostra_fft[n_bands_fft - 1] = np.sum(band ** 2)  # energy
    return amostra_fft
