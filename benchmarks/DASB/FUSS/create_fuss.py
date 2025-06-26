import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm

def create_silent_audio(reference_path, target_path):
    """
    Create a silent audio file with the same length and sampling rate as the reference audio.

    Args:
        reference_path (str): Path to the reference audio file.
        target_path (str): Path where the silent audio will be saved.
    """
    # Read the reference audio to get sampling rate and length
    data, samplerate = sf.read(reference_path)
    silent_audio = np.zeros_like(data)

    # Save the silent audio
    sf.write(target_path, silent_audio, samplerate)

def create_mixture_audio(directory, required_files, output_path):
    """
    Create a mixture audio file that is a linear mix of all existing audio files in the directory.

    Args:
        directory (str): Path to the directory containing the audio files.
        required_files (list): List of required audio file names.
        output_path (str): Path where the mixture audio will be saved.
    """
    mixture = None
    samplerate = None

    for file in required_files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            data, sr = sf.read(file_path)
            if mixture is None:
                mixture = np.zeros_like(data, dtype=np.float32)
                samplerate = sr
            mixture += data

    if mixture is not None and samplerate is not None:
        # Normalize the mixture to prevent clipping
        #mixture = mixture / len(required_files)
        sf.write(output_path, mixture, samplerate)

def ensure_audio_files(directory):
    """
    Ensure all required audio files exist in a directory. If not, create silent versions of them.

    Args:
        directory (str): Path to the directory containing the audio files.
    """
    required_files = [
        "background0_sound.wav",
        "foreground0_sound.wav",
        "foreground1_sound.wav",
        "foreground2_sound.wav",
    ]

    # Full paths to the required files
    required_paths = {file: os.path.join(directory, file) for file in required_files}

    # Check if 'background0_sound.wav' exists
    background_path = required_paths["background0_sound.wav"]
    if not os.path.exists(background_path):
        print(f"Error: {background_path} is missing. Cannot proceed.")
        return

    # Ensure other files exist, creating silent versions if necessary
    for file, path in required_paths.items():
        if not os.path.exists(path):
            #print(f"{file} is missing. Creating a silent version.")
            create_silent_audio(background_path, path)

    # Create the mixture audio file
    mixture_path = os.path.join(directory, "mixture.wav")
    create_mixture_audio(directory, required_files, mixture_path)

def process_directories(root_directory):
    """
    Walk through each subdirectory and ensure required audio files exist and create mixture files.

    Args:
        root_directory (str): Path to the root directory of the FUSS eval set.
    """
    for subdir, _, _ in tqdm(os.walk(root_directory)):
        ensure_audio_files(subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure audio files and create mixture files in each subdirectory.")
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory of the FUSS eval set."
    )

    args = parser.parse_args()
    root_dir = args.root_dir
    process_directories(root_dir)
