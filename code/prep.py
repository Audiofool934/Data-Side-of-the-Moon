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
    
    root_path="data/audio/GTZAN"
    save_path="data/spec/GTZAN"
    
    spec_convert_folder(root_path, save_path, formats)