import librosa
import numpy as np
import pandas as pd
import os, re


def normalize_log_mel_spec(log_mel_spec, max=1, min=0):
    """
    Normalize the log mel spectrogram to the range (-1, 1).
    """
    norm_spec = (log_mel_spec - np.min(log_mel_spec)) / (
        np.max(log_mel_spec) - np.min(log_mel_spec)
    )
    norm_spec = norm_spec * (max - min) + min
    return norm_spec

def convert_to_spec(path, save_path):

    y, sr = librosa.load(path, duration=15)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=256, n_fft=2048, hop_length=512
    )
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # make dimensions of the array even 128x1292
    # log_mel_spec = np.resize(log_mel_spec,(256,1292))

    # print(log_mel_spec.shape)

    # librosa.display.specshow(log_mel_spec, sr=sr, hop_length=512)
    # plt.show()

    # print(log_mel_spec)

    log_mel_spec = normalize_log_mel_spec(log_mel_spec)

    # log_mel_spec_norm=log_mel_spec_norm[:,:640]

    np.save(save_path, log_mel_spec, allow_pickle=True)


def create_directory_structure(root_path, save_path, file_path, formats):
    relative_path = os.path.relpath(file_path, root_path)
    target_dir = os.path.join(save_path, os.path.dirname(relative_path))
    os.makedirs(target_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    
    for ext in formats:
        if base_name.endswith(ext):
            base_name = base_name.replace(ext, '.npy')
            break
            
    return os.path.join(target_dir, base_name)

def get_all_files_paths(root_path, formats=[".mp3",".wav"]):
    audio_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in formats):
                audio_files.append(os.path.join(dirpath, filename))
    return audio_files

def split_audio(y, sr, segment_duration=15):
    segment_length = segment_duration * sr
    segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
    if len(segments[-1]) < segment_length:
        segments = segments[:-1]  # Remove the last segment if it's shorter than the segment length
    return segments

def extract_segment(input_file, start_time_ms, duration_ms=30000):
    # Load the audio file
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(input_file)
    
    # Calculate start and end time of the segment
    start_ms = max(0, start_time_ms - duration_ms // 2)
    end_ms = min(len(audio), start_time_ms + duration_ms // 2)
    
    # Extract the segment
    segment = audio[start_ms:end_ms]
    
    return segment

def save_segment(segment, output_file):
    segment.export(output_file, format="mp3")

# # Usage
# input_file = "test/Whole Lotta Love.mp3"
# output_file = "test/Whole Lotta Love_cut.mp3"
# start_time_ms = 60000  # Start time in milliseconds

# segment = extract_segment(input_file, start_time_ms)
# save_segment(segment, output_file)

def explore(file_path):
    # Load the numpy array from the uploaded file
    data = np.load(file_path)

    data_shape = data.shape
    data_dtype = data.dtype

    print(data_shape, data_dtype)

    print(type(data))

    print(data[:5])
    
# path="datasets/my_dataset_640"

# files=get_all_files_paths(path, "npy")

# for file in files[:5]:
#     arr=np.load(file)
#     print(arr.shape)
#     # arr=arr[:,:640]

#     # np.save(f"{file}", arr)

def read_h5_as_dataframe(filepath):
    try:
        df = pd.read_hdf(filepath, key="extract")
        print("Data loaded successfully")
        return df
    except Exception as e:
        print(f"Cannot load data. An error occurred: {e}")
        

######## meta_extract ########

def extract_info(path):
    # Split the path into components
    components = path.split('/')

    # Handle cases where there might be an additional directory like "Disc 1"
    if not ord('0') <= ord(components[-2].strip()[0]) <= ord('9'):
        disc_info = components[-2].strip()
        track_info_index = 1
    else:
        disc_info = ""
        track_info_index = 0

    # Extract the artist name
    artist = components[-4-track_info_index].strip()
    
    # Extract the release year and album name
    year_album = components[-2-track_info_index].strip()
    year, album = year_album.split(" ", maxsplit=1)
    if album[0] == '-':
        album = album.split('-', 1)[-1].strip()

    # Extract the track number and name
    components[-1] = components[-1].strip()
    for i in range(len(components[-1])):
        if not ord('0') <= ord(components[-1][i]) <= ord('9'):
            index = i
            break
    track_number = components[-1][:index].strip()
    track_name = components[-1][index:].strip()
    for i in range(len(track_name)):
        if track_name[i].isalpha() or track_name[i].isdigit():
            index = i
            break
    track_name = track_name[index:].strip().rsplit('.', 1)[0]


    if disc_info:
        track_number = f"{disc_info}-{track_number}"
    return artist, year, album, track_number, track_name

def extract_song_name(file_path):
    """从文件路径中提取歌曲名称，忽略前面的音轨号"""
    base_name = os.path.basename(file_path)
    # 使用正则表达式忽略前面的音轨号（包括可能的空格和分隔符）
    song_name = re.sub(r'^\d+\s*[-.]*\s*', '', base_name)
    song_name = os.path.splitext(song_name)[0]
    return song_name