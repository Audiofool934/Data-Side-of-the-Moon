import pandas as pd
import h5py
from mfcc import compute_mfcc
from utility import get_all_files_paths, extract_info
from encode_songs import encode_song


def extract_and_save(root_path, savepath, model_path, Pieces=None, weight=None, Mode='MFCC'):

    audio_paths = get_all_files_paths(root_path, ['.mp3'])
    artist_list = []
    year_list = []
    album_list = []
    track_number_list = []
    song_list = []
    extract_list = []

    with h5py.File(savepath, "w") as h5file:

        for path in audio_paths:
            if Mode == 'MFCC':
                mfcc_mean = compute_mfcc(path, Pieces=Pieces, weight=weight)
                extract_list.append(mfcc_mean)
                
            elif Mode == 'AE':
                extract_list.append(encode_song(path, model_path=model_path))
            
            artist, year, album, track_number, track_name = extract_info(path)
            artist_list.append(artist)
            year_list.append(year)
            album_list.append(album)
            track_number_list.append(track_number)
            song_list.append(track_name)

    df = pd.DataFrame({
        "Artist": artist_list,
        "Year": year_list,
        "Album": album_list,
        "Track Number": track_number_list,
        "Song": song_list,
        Mode: extract_list,
    })

    df.to_hdf(savepath, key="extract", mode="w")


if __name__ == "__main__":

    Mode = 'AE'
    # Mode = "MFCC"
    music_path = 'data/audio/Pink Floyd/Discography (1967-2014)'
    save_path = f'database/{Mode}.h5'
    model_path="models/Echoes_128/encoder.pth"
    extract_and_save(music_path, save_path,model_path=model_path, Pieces=None, weight=None, Mode=Mode)
