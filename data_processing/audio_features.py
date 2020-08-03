import numpy as np
import scipy.io.wavfile as wav
import librosa
import matplotlib.pyplot as plt
import sys
import os
import math

def extract_melspec(audio_dir, files, destpath, fps):

    for f in files:
        file = os.path.join(audio_dir, f + '.wav')
        outfile = destpath + '/' + f + '.npy'
        
        print('{}\t->\t{}'.format(file,outfile))
        fs,X = wav.read(file)
        X = X.astype(float)/math.pow(2,15)
        
        assert fs%fps == 0
        
        hop_len=int(fs/fps)
        
        n_fft=int(fs*0.13)
        C = librosa.feature.melspectrogram(y=X, sr=fs, n_fft=2048, hop_length=hop_len, n_mels=27, fmin=0.0, fmax=8000)
        C = np.log(C)
        print("fs: " + str(fs))
        print("hop_len: " + str(hop_len))
        print("n_fft: " + str(n_fft))
        print(C.shape)
        print(np.min(C),np.max(C))
        np.save(outfile,np.transpose(C))
