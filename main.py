import librosa
import soundfile as sf
from scipy.io import wavfile
import scipy.io
from pysndfx import AudioEffectsChain as AEC
import numpy as np
import matplotlib.pyplot as plt

def load_file(f):
    return librosa.core.load(f, sr=12000)

def save_file(f, sr, wav):
    sf.write(f, wav, sr)

def remove_sidesignals(sr, wav):
    N = len(wav)-1
    Y_k = np.fft.fft(wav)[0:int(N/2)]/N # FFT function from numpy
    Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
    Pxx = np.abs(Y_k) # be sure to get rid of imaginary part
    f = sr*np.arange(N/2)/N
    
    poi = np.argmax(f)
    print(poi)
    fx = AEC().equalizer(frequency=525,q=1000,db=20)

    wav = fx(wav)

    return wav



inp, samplerate = load_file("input/01.wav")
out = remove_sidesignals(samplerate, inp)
save_file("output/01.wav", samplerate, out)
print(out)