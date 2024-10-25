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

    # convert_to_spec("datasets/testsongs/Queen – Bohemian Rhapsody.mp3","/Users/audiofool/Desktop/reimplement/datasets")

    # build_dataset()
    
    formats = ['.mp3', '.wav', '.flac'] 
    
    root_path="data/audio/fma_small_by_genre/genre_unknown"
    save_path="data/spec/fma_small/genre_unknown"
    
    # Experimental 有一个错？
    # Error processing file data/audio/fma_small_by_genre/genre_unknown/108925.mp3: 
    # ^[[D[src/libmpg123/parse.c:do_readahead():1099] warning: Cannot read next header, a one-frame stream? Duh...
    # Error processing file data/audio/fma_small_by_genre/genre_unknown/099134.mp3: 
    
    spec_convert_folder(root_path, save_path, formats)