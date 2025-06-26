import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from copy import copy
from concurrent.futures import ProcessPoolExecutor

def apply_random_gain(audio, min_gain=0.25, max_gain=1.25):
    """
    Apply a random gain to a numpy array representing an audio signal.

    Args:
        audio (numpy.ndarray): Input audio signal.
        min_gain (float): Minimum gain value.
        max_gain (float): Maximum gain value.

    Returns:
        numpy.ndarray: Audio signal with random gain applied.
    """
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain

def ensure_audio_files(directory, required_files):
    """
    Ensure all required audio files exist in a directory.
    """
    required_paths = {file: os.path.join(directory, file) for file in required_files}
    if not all(os.path.exists(p) for p in required_paths.values()):
        print(f"Error: Missing files in {directory}. Cannot proceed.")
        return False
    return True

def random_chunk_indices(total_samples, chunk_samples, num_chunks):
    """
    Generate random start indices for chunks within the range of the audio length.
    Ensures chunks do not exceed the total length.
    """
    max_start = total_samples - chunk_samples
    if max_start <= 0:
        return [0] * num_chunks  # Only one possible chunk if audio is shorter than chunk size
    return np.random.randint(0, max_start + 1, size=num_chunks)

def process_track(split, track, track_path, target_dir, chunk_size, num_chunks, required_files):
    """
    Process a single track by randomly sampling chunks and saving them.
    """
    if not ensure_audio_files(track_path, required_files):
        return
    
    audio_data = {}
    sample_rate = None
    total_samples = None
    
    # Load all required files and convert to mono if needed
    for file in required_files:
        file_path = os.path.join(track_path, file)
        audio, sr = sf.read(file_path)
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
        if sample_rate is None:
            sample_rate = sr
        if total_samples is None:
            total_samples = len(audio)
        audio_data[file] = audio
    
    chunk_samples = int(chunk_size * sample_rate)
    start_indices = random_chunk_indices(total_samples, chunk_samples, num_chunks)
    
    # Save randomly sampled chunks
    for i, start in enumerate(start_indices):
        end = start + chunk_samples
        chunk_sum = np.zeros(chunk_samples)  # Initialize for mixture
        for file in required_files:
            chunk = copy(audio_data[file][start:end])
            if not split == 'eval':
                chunk = apply_random_gain(chunk) 
            chunk_sum += chunk  # Add to mixture
            
            new_track_name = f"{track.replace(' ', '_')}_chunk{i:02d}"
            save_path = os.path.join(target_dir, split, new_track_name)
            os.makedirs(save_path, exist_ok=True)
            sf.write(os.path.join(save_path, file), chunk, sample_rate)
        
        # Save the computed mixture
        sf.write(os.path.join(save_path, "mixture.wav"), chunk_sum, sample_rate)

def process_tracks(root_dir, target_dir, chunk_size, num_chunks, max_workers=4):
    """
    Processes the dataset by randomly sampling chunks from each track using parallel processing.
    Args:
        root_dir (str): Root directory containing the dataset.
        target_dir (str): Target directory to save the processed chunks.
        chunk_size (int): Size of each chunk in seconds.
        num_chunks (int): Number of random chunks per track.
        max_workers (int): Maximum number of parallel workers.
    """
    required_files = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    tasks = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for split in ['train']:#os.listdir(root_dir):
            split_path = os.path.join(root_dir, split)
            if not os.path.isdir(split_path):
                continue
            
            for track in os.listdir(split_path):
                track_path = os.path.join(split_path, track)
                if not os.path.isdir(track_path):
                    continue
                tasks.append(executor.submit(process_track, split, track, track_path, target_dir, chunk_size, num_chunks, required_files))
    
    # Wait for all tasks to complete
    for task in tqdm(tasks, desc="Processing tracks"):
        task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random audio chunks from tracks in parallel.")

    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory containing source audio tracks."
    )

    parser.add_argument(
        "target_dir",
        type=str,
        help="Path to the directory where processed chunks will be saved."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5,
        help="Size of each audio chunk in seconds (default: 5)"
    )

    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1000,
        help="Number of random chunks to extract per track (default: 1000)"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of parallel workers (default: 32)"
    )

    args = parser.parse_args()

    process_tracks(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        chunk_size=args.chunk_size,
        num_chunks=args.num_chunks,
        max_workers=args.max_workers
    )

