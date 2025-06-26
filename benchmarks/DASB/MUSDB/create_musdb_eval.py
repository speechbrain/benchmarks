import os
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def ensure_audio_files(directory, required_files):
    """
    Ensure all required audio files exist in a directory.
    """
    required_paths = {file: os.path.join(directory, file) for file in required_files}
    if not all(os.path.exists(p) for p in required_paths.values()):
        print(f"Error: Missing files in {directory}. Cannot proceed.")
        return False
    return True

def process_track(split, track, track_path, target_dir, chunk_size, required_files):
    """
    Process a single track by sequentially partitioning it into non-overlapping chunks.
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
    num_chunks = total_samples // chunk_samples
    
    # Save sequentially sampled chunks
    for i in range(num_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_sum = np.zeros(chunk_samples)  # Initialize for mixture
        
        new_track_name = f"{track.replace(' ', '_')}_chunk{i:02d}"
        save_path = os.path.join(target_dir, split, new_track_name)
        os.makedirs(save_path, exist_ok=True)
        
        for file in required_files:
            chunk = audio_data[file][start:end]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunk_sum += chunk  # Add to mixture
            sf.write(os.path.join(save_path, file), chunk, sample_rate)
        
        # Save the computed mixture
        sf.write(os.path.join(save_path, "mixture.wav"), chunk_sum, sample_rate)

def process_tracks(root_dir, target_dir, chunk_size, max_workers=4):
    """
    Processes the dataset by sequentially partitioning tracks using parallel processing.
    Args:
        root_dir (str): Root directory containing the dataset.
        target_dir (str): Target directory to save the processed chunks.
        chunk_size (int): Size of each chunk in seconds.
        max_workers (int): Maximum number of parallel workers.
    """
    required_files = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    tasks = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for split in ['validation', 'eval']:
            split_path = os.path.join(root_dir, split)
            if not os.path.isdir(split_path):
                continue
            
            for track in os.listdir(split_path):
                track_path = os.path.join(split_path, track)
                if not os.path.isdir(track_path):
                    continue
                tasks.append(executor.submit(process_track, split, track, track_path, target_dir, chunk_size, required_files))
    
    # Wait for all tasks to complete
    for task in tqdm(tasks, desc="Processing tracks"):
        task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio tracks in parallel with chunking.")

    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory containing MUSDB source audio tracks."
    )

    parser.add_argument(
        "target_dir",
        type=str,
        help="Path to the target directory where processed chunks will be saved."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5,
        help="Chunk size in seconds (default: 5)"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers (default: 8)"
    )

    args = parser.parse_args()

    process_tracks(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers
    )
