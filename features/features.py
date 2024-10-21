import os
import numpy as np
import librosa
import pandas as pd

def extract_features(audio_file):
    # Load the audio file
    signal, sr = librosa.load(audio_file, sr=None)

    # Extract LFCC features
    lfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    lfcc_mean = np.mean(lfcc, axis=1)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return lfcc_mean, mfcc_mean

def process_audio_files(audio_directory, output_directory, num_samples=6000):
    audio_files = [f for f in os.listdir(audio_directory) if f.endswith('.flac')][:num_samples]
    
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for audio_file in audio_files:
        file_path = os.path.join(audio_directory, audio_file)
        
        lfcc, mfcc = extract_features(file_path)

        # Save features as CSV
        base_filename = os.path.splitext(audio_file)[0]
        lfcc_file = os.path.join(output_directory, f"{base_filename}_LFCC.csv")
        mfcc_file = os.path.join(output_directory, f"{base_filename}_MFCC.csv")

        pd.DataFrame(lfcc).to_csv(lfcc_file, header=False, index=False)
        pd.DataFrame(mfcc).to_csv(mfcc_file, header=False, index=False)

if __name__ == "__main__":
    audio_directory = r"C:\Users\vibha srinivasan\Desktop\spoofing system\VSD_ASV\audioflac"
    output_directory = r"C:\Users\vibha srinivasan\Desktop\spoofing system\VSD_ASV\features_extracted"
    process_audio_files(audio_directory, output_directory, num_samples=6000)
