import numpy as np
# import librosa #has to do this cause librosa is not supported on my server
import python_speech_features
from scipy.io import wavfile
from scipy import signal

# def load_wav(audio_fn, sr = 16000):
#     y, sr = librosa.load(audio_fn, sr = sr, mono=True)    
#     return y, sr

def load_wav_old(audio_fn, sr = 16000):
    sample_rate, sig = wavfile.read(audio_fn)
    if sample_rate != sr:
        result = int((sig.shape[0]) / sample_rate * sr)
        x_resampled = signal.resample(sig, result)
        x_resampled = x_resampled.astype(np.float64)
        return x_resampled, sr
    
    sig = sig / (2**15)
    return sig, sample_rate

def get_mfcc(audio_fn, eps=1e-6, fps=25, sr=16000, n_mfcc=64, win_size=None):
    raise NotImplementedError
    '''
    # y, sr = load_wav(audio_fn, sr=sr)

    # if win_size is None:
    #     hop_len=int(sr / fps)
    # else:
    #     hop_len=int(sr/ win_size)
        
    # n_fft=2048 

    # C = librosa.feature.mfcc(
    #     y = y,
    #     sr = sr,
    #     n_mfcc = n_mfcc,
    #     hop_length = hop_len,
    #     n_fft = n_fft
    # )

    # if C.shape[0] == n_mfcc:
    #     C = C.transpose(1, 0)
    
    # return C
    '''
    
def get_melspec(audio_fn, eps=1e-6, fps = 25, sr=16000, n_mels=64):
    raise NotImplementedError
    '''
    # y, sr = load_wav(audio_fn=audio_fn, sr=sr)
    
    # hop_len = int(sr / fps) 
    # n_fft = 2048

    # C = librosa.feature.melspectrogram(
    #     y = y, 
    #     sr = sr, 
    #     n_fft=n_fft, 
    #     hop_length=hop_len, 
    #     n_mels = n_mels, 
    #     fmin=0, 
    #     fmax=8000)
    

    # mask = (C == 0).astype(np.float)
    # C = mask * eps + (1-mask) * C

    # C = np.log(C)
    # #wierd error may occur here
    # assert not (np.isnan(C).any()), audio_fn
    # if C.shape[0] == n_mels:
    #     C = C.transpose(1, 0)

    # return C 
    '''

def extract_mfcc(audio,sample_rate=16000):
    mfcc = zip(*python_speech_features.mfcc(audio,sample_rate, numcep=64, nfilt=64, nfft=2048, winstep=0.04))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc

def get_mfcc_psf(audio_fn, eps=1e-6, fps=25, sr=16000, n_mfcc=64, win_size=None):
    y, sr = load_wav_old(audio_fn, sr=sr)

    if win_size is None:
        hop_len=int(sr / fps)
    else:
        hop_len=int(sr/ win_size)
        
    n_fft=2048 

    #hard coded for 25 fps
    C = python_speech_features.mfcc(y, sr, numcep=n_mfcc, nfilt=n_mfcc, nfft=n_fft, winstep=0.04)

    # if C.shape[0] == n_mfcc:
    #     C = C.transpose(1, 0)
    
    return C

def get_mfcc_old(wav_file):
    sig, sample_rate = load_wav_old(wav_file)
    mfcc = extract_mfcc(sig)
    return mfcc

if __name__ == '__main__':
    audio_fn = '../sample_audio/clip000028_tCAkv4ggPgI.wav'
    
    C = get_mfcc_psf(audio_fn)
    print(C.shape)

    C_2 = get_mfcc_librosa(audio_fn)
    print(C.shape)

    print(C)
    print(C_2)
    print((C == C_2).all())
    # print(y.shape, sr)
    # mel_spec = get_melspec(audio_fn)
    # print(mel_spec.shape)
    # mfcc = get_mfcc(audio_fn, sr = 16000)
    # print(mfcc.shape)
    # print(mel_spec.max(), mel_spec.min())
    # print(mfcc.max(), mfcc.min())