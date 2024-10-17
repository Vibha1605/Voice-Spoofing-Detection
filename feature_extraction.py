import os
import random
import pandas as pd
import librosa
import numpy as np

# Load audio files
def load_audio_files(directory, num_files=6000):
    all_files = [f for f in os.listdir(directory) if f.endswith('.flac')]
    selected_files = random.sample(all_files, num_files)
    return selected_files

# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    lfcc = librosa.feature.lfcc(y=y, sr=sr)  # Check if this function is available in your librosa version
    return mfcc, lfcc

# Main execution
audio_directory = r'C:\Users\Serilda\Desktop\Final Year Project\ASVspoof2021_LA_eval\audioflac'
selected_files = load_audio_files(audio_directory)

features = []
for file in selected_files:
    mfcc, lfcc = extract_features(os.path.join(audio_directory, file))
    combined = np.concatenate((mfcc, lfcc), axis=0)
    features.append(combined)

# Save to CSV
df = pd.DataFrame(features)
df.to_csv('extracted_features.csv', index=False)
