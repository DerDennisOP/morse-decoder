import librosa
import soundfile as sf
from scipy.io import wavfile
import scipy.io
from scipy.signal import butter, lfilter, firwin
from scipy.fft import fft, fftfreq
from pysndfx import AudioEffectsChain as AEC
import numpy as np
import matplotlib.pyplot as plt

import morsetotext as mtt

def load_file(f):
    return librosa.core.load(f, sr=12000)

def save_file(f, sr, wav):
    sf.write(f, wav, sr)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def remove_sidesignals(sr, wav, freqency=None, bandwidth=50):
    if freqency is None or freqency == 0:
        yf = np.abs(fft(wav))
        xf = np.abs(fftfreq(len(wav), 1 / sr))
        poi = xf[np.argmax(yf)]
    else:
        poi = freqency

    print(f"Frequency: {poi}")

    wav = butter_bandpass_filter(wav, poi-bandwidth/2, poi+bandwidth/2, sr, 6)

    return wav

def get_high(wav, sr, threshold=0.03):    
    wav = np.abs(wav)
    # wav = butter_lowpass_filter(wav, threshold, sr, order=4)*(10**18)
    y = []
    avl = []
    tmp = 0

    taps = firwin(100,80/(sr*0.5),window="blackmanharris", nyq=sr*0.5)
    wav = lfilter(taps, 1.0, wav)

    plt.plot(wav)
    plt.show()
    
    for x in range(len(wav)):
        # print(wav[x])
        if wav[x] > threshold:
            y.append(1.)
            tmp += 1
        else:
            y.append(0.)
            if len(y) > 2 and y[-2] == 1.:
                avl.append([tmp, x])
                tmp = 0

    # plt.plot(y)
    # plt.show()
    return avl
    
def get_period(l):
    y = []
    tmp = []
    
    avl = 0
    lshort = 1800

    if len(l) > 0:

        for x in l:
            avl += x[0]
    
        avl = round(avl / len(l))

        for x in range(len(l)):
            if l[x][0] <= avl-1000:
                lshort = l[x][0]*2
                tmp.append(0)
            else:
                tmp.append(1)
        
            if len(l) > x+1:
                if l[x+1][1] - l[x][1] >= lshort*1 + l[x+1][0]:
                    y.append(tmp)
                    tmp = []
                
                if l[x+1][1] - l[x][1] >= lshort*2 + l[x+1][0]:
                    y.append(' ')
        
        if len(tmp) > 0:
            y.append(tmp)

    return y






inp, samplerate = load_file("input/01.wav")
out = remove_sidesignals(samplerate, inp, freqency=0, bandwidth=50)
length = get_high(out, samplerate)
out = get_period(length)
out = mtt.convert(out)
# print(out)
# save_file("output/02.wav", samplerate, out)
print(out)