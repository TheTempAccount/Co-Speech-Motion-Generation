import math
from librosa.core import audio
import numpy as np
import python_speech_features
from scipy.io import wavfile
from scipy import signal
import math
import librosa

def load_wav(audio_fn, sr = 16000):
    y, sr = librosa.load(audio_fn, sr = sr, mono=True)    
    return y, sr

def get_mfcc(audio_fn, eps=1e-6, fps=25, sr=16000, n_mfcc=64, win_size=None):
    y, sr = load_wav(audio_fn, sr=sr)

    if win_size is None:
        hop_len=int(sr / fps)
    else:
        hop_len=int(sr/ win_size)
        
    n_fft=2048 

    C = librosa.feature.mfcc(
        y = y,
        sr = sr,
        n_mfcc = n_mfcc,
        hop_length = hop_len,
        n_fft = n_fft
    )

    if C.shape[0] == n_mfcc:
        C = C.transpose(1, 0)
    
    return C
    

def get_melspec(audio_fn, eps=1e-6, fps = 25, sr=16000, n_mels=64):
    y, sr = load_wav(audio_fn=audio_fn, sr=sr)
    
    hop_len = int(sr / fps) 
    n_fft = 2048

    C = librosa.feature.melspectrogram(
        y = y, 
        sr = sr, 
        n_fft=n_fft, 
        hop_length=hop_len, 
        n_mels = n_mels, 
        fmin=0, 
        fmax=8000)
    

    mask = (C == 0).astype(np.float)
    C = mask * eps + (1-mask) * C

    C = np.log(C)
    if C.shape[0] == n_mels:
        C = C.transpose(1, 0)

    return C 

if __name__ == '__main__':
    audio_fn = 'sample_audio/jon.wav'
    y, sr = load_wav(audio_fn=audio_fn, sr=16000)
    print(y.shape, sr)
    mel_spec = get_melspec(audio_fn)
    print(mel_spec.shape)
    mfcc = get_mfcc(audio_fn, sr = 16000)
    print(mfcc.shape)
    print(mel_spec.max(), mel_spec.min())
    print(mfcc.max(), mfcc.min())