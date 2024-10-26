from utility import get_all_files_paths, create_directory_structure, convert_to_spec

def spec_convert_song(root_path, save_path, file, formats=[".mp3",".wav"]):
    spec_save_path = create_directory_structure(root_path, save_path, file, formats)
    convert_to_spec(file, spec_save_path)

def spec_convert_folder(root_path, save_path, formats=[".mp3",".wav"]):
    files = get_all_files_paths(root_path, formats)

    for file in files:
        spec_convert_song(root_path, save_path, file, formats)

if __name__ == "__main__":

    FRAME_SIZE = 512
    HOP_LENGTH = 256
    SAMPLE_RATE = 22050
    MONO = True
    
    formats = ['.mp3', '.wav', '.flac'] 
    
    root_path="data/audio/fma_small"
    save_path="data/spec/fma_small"
    
    spec_convert_folder(root_path, save_path, formats)
    
    '''
    [src/libmpg123/layer3.c:INT123_do_layer3():1771] error: part2_3_length (3360) too large for available bit count (3240)
    [src/libmpg123/layer3.c:INT123_do_layer3():1771] error: part2_3_length (3328) too large for available bit count (3240)
    [src/libmpg123/layer3.c:INT123_do_layer3():1801] error: dequantization failed!
    [src/libmpg123/parse.c:do_readahead():1099] warning: Cannot read next header, a one-frame stream? Duh...
    /home/chuangyan/miniconda3/envs/art2mus/lib/python3.10/site-packages/librosa/util/decorators.py:88: UserWarning: PySoundFile failed. Trying audioread instead.
    return f(*args, **kwargs)
    Error processing file data/audio/fma_small/099/099134.mp3: 
    [src/libmpg123/layer3.c:INT123_do_layer3():1801] error: dequantization failed!
    [src/libmpg123/parse.c:do_readahead():1099] warning: Cannot read next header, a one-frame stream? Duh...
    Error processing file data/audio/fma_small/108/108925.mp3: 
    /home/chuangyan/Data-Side-of-the-Moon/code/utility.py:11: RuntimeWarning: invalid value encountered in divide
    norm_spec = (log_mel_spec - np.min(log_mel_spec)) / (
    Note: Illegal Audio-MPEG-Header 0x00000000 at offset 33361.
    Note: Trying to resync...
    Note: Skipped 1024 bytes in input.
    [src/libmpg123/parse.c:wetwork():1365] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
    Error processing file data/audio/fma_small/098/098565.mp3: 
    Note: Illegal Audio-MPEG-Header 0x00000000 at offset 22401.
    Note: Trying to resync...
    Note: Skipped 1024 bytes in input.
    [src/libmpg123/parse.c:wetwork():1365] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
    Error processing file data/audio/fma_small/098/098567.mp3: 
    [src/libmpg123/layer3.c:INT123_do_layer3():1801] error: dequantization failed!
    Note: Illegal Audio-MPEG-Header 0x00000000 at offset 63168.
    Note: Trying to resync...
    Note: Skipped 1024 bytes in input.
    [src/libmpg123/parse.c:wetwork():1365] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
    Error processing file data/audio/fma_small/098/098569.mp3: 
    [src/libmpg123/parse.c:do_readahead():1099] warning: Cannot read next header, a one-frame stream? Duh...
    Error processing file data/audio/fma_small/133/133297.mp3: 
    [src/libmpg123/layer3.c:INT123_do_layer3():1841] error: dequantization failed!
    '''