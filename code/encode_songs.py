import numpy as np
import librosa
from utility import normalize_log_mel_spec, get_all_files_paths, split_audio
from encoder import load_encoder, encode_data

def calculate_volume(segment):
    return np.sqrt(np.mean(segment**2))

def encode_segments(encoder, segments, sr, avg=False):
    encoded_segments = []
    volumes = []

    for segment in segments:
        if not avg:
            volume = calculate_volume(segment)
            volumes.append(volume)
        
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=256, n_fft=2048, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec_norm = normalize_log_mel_spec(log_mel_spec)
        
        encoded_segments.append(encode_data(encoder, log_mel_spec_norm))

    if avg:
        return np.mean(encoded_segments, axis=0)
    else:
        volumes = np.array(volumes)
        normalized_volumes = volumes / volumes.sum()  # Normalize volumes to sum to 1
        weighted_encoded_segments = np.array([encoded_segments[i] * normalized_volumes[i] for i in range(len(encoded_segments))])
        return np.sum(weighted_encoded_segments, axis=0)    


def encode_song(file_path, model_path = "models/encoder.pth", encoded_space_dim = 128):

    encoder = load_encoder(model_path, encoded_space_dim)

    y, sr = librosa.load(file_path)
    segments = split_audio(y, sr, segment_duration=15)

    return encode_segments(encoder, segments, sr, avg=False)

def encode_songs_save(file_path, save_path, file_format="mp3", model_path = "models/encoder.pth", encoded_space_dim = 128):
    
    encoder = load_encoder(model_path, encoded_space_dim)

    audiofiles = get_all_files_paths(file_path, file_format)
    
    for file in audiofiles:
        y, sr = librosa.load(file)

        segments = split_audio(y, sr, segment_duration=15)

        encoded_average = encode_segments(encoder, segments, sr, avg=False)
        
        name=file.split("/")[-1]
        # album=file.split("/")[-2]
        # path=f"datasets/songs/{album}"
        # create_directory(path)
        # np.save(f"{path}/{name}.npy", encoded_average)
        np.save(f"{save_path}/{name}.npy", encoded_average)

if __name__=="__main__":

    model_path="models/Echoes_128/encoder.pth"
    file_path="data/audio/testsongs"
    file_format="mp3"
    save_path="data/encoded"
    
    encode_songs_save(file_path=file_path, save_path=save_path, model_path=model_path)