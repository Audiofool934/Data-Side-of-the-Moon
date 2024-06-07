import librosa
import numpy as np

def normalize(audio):
    audio = librosa.util.normalize(audio)
    return audio
def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

def compute_mfcc(audio_path, n_mfcc=20, Pieces=None, weight=None):
    y, sr = load_audio(audio_path)
    Y = None
    if Pieces is None:
        Y = y
    else:
        duration = librosa.get_duration(y=y, sr=sr)
        for piece in Pieces:
            phi = piece[0]
            length = piece[1]
            center_time = phi * duration
            start_time_segment = max(0, center_time - length/2)
            end_time_segment = min(duration, center_time + length/2)
            start_sample_segment = int(start_time_segment * sr)
            end_sample_segment = int(end_time_segment * sr)
            segment = y[start_sample_segment:end_sample_segment]
            if Y is None:
                Y = segment
            else:
                Y = np.concatenate((Y, segment))
    mfcc = librosa.feature.mfcc(y=Y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    if weight is None:
        return mfcc_mean
    else:
        mfccs_segment = mfcc_mean
        mfccs_global = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        mfccs_global_normalized = (mfccs_global - np.mean(mfccs_global)) / np.std(mfccs_global)
        mfccs_segment_normalized = (mfccs_segment - np.mean(mfccs_segment)) / np.std(mfccs_segment)
        mfccs_combined = mfccs_global_normalized + mfccs_segment_normalized * weight
        return mfccs_combined
