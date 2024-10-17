import os
import librosa
import numpy as np
import pandas as pd
from scipy.fftpack import dct

def extract_mfcc(speech, Fs, Window_Length, NFFT, No_Filter):
    # Pre-emphasis on the speech signal
    speech[1:] = speech[1:] - 0.97 * speech[:-1]  
    
    if np.all(speech == 0):  # Check if the sample is all zeros
        print("Warning: Sample contains only zeros.")
        return None  # Skip this sample

    frame_length_in_samples = int((Fs / 1000) * Window_Length)
    hop_length = frame_length_in_samples // 2  # Overlap of 50%
    
    framedspeech = librosa.util.frame(speech, frame_length=frame_length_in_samples, hop_length=hop_length).T

    w = np.hamming(frame_length_in_samples)
    y_framed = framedspeech * w

    fr_all = np.abs(np.fft.fft(y_framed, NFFT))**2
    fr_all = fr_all[:, :NFFT // 2 + 1]  
    
    mel_filters = librosa.filters.mel(sr=Fs, n_fft=NFFT, n_mels=No_Filter, fmin=0, fmax=Fs // 2)
    mel_spectrum = np.dot(fr_all, mel_filters.T)
    
    log_mel_spectrum = np.log10(np.maximum(mel_spectrum, np.finfo(float).eps))   
    mfccs = dct(log_mel_spectrum, type=2, axis=1, norm='ortho')[:, :No_Filter]
    
    return mfccs

def extract_lfcc(speech, Fs, Window_Length, NFFT, No_Filter):
    speech = np.append(speech[0], speech[1:] - 0.97 * speech[:-1])
    
    if np.all(speech == 0):
        print("Warning: Sample contains only zeros.")
        return None  # Skip this sample

    frame_length_in_samples = int((Fs / 1000) * Window_Length)
    hop_length = frame_length_in_samples // 2
    framedspeech = librosa.util.frame(speech, frame_length=frame_length_in_samples, hop_length=hop_length).T
    w = np.hamming(frame_length_in_samples)
    y_framed = framedspeech * w
    
    fr_all = np.abs(np.fft.fft(y_framed, NFFT))**2
    fa_all = fr_all[:, :NFFT // 2 + 1]
    
    f = (Fs / 2) * np.linspace(0, 1, NFFT // 2 + 1)
    filter_bandwidths = np.linspace(min(f), max(f), No_Filter + 2)
    filterbank = np.zeros((NFFT // 2 + 1, No_Filter))
    for i in range(No_Filter):
        filterbank[:, i] = np.maximum(0, 1 - np.abs((f - filter_bandwidths[i]) / (filter_bandwidths[i + 1] - filter_bandwidths[i])))

    filbanksum = np.dot(fa_all, filterbank)

    if np.all(filbanksum == 0):
        print("Warning: Sample has all-zero filter bank sum.")
        return None

    t = dct(np.log10(filbanksum + np.finfo(float).eps), type=2, axis=1, norm='ortho')[:, :No_Filter]

    return t.T

def extract_features(samples, Fs, Window_Length, NFFT, No_Filter):
    all_mfccs, all_lfccs = [], []
    
    for idx, speech in enumerate(samples):
        mfccs = extract_mfcc(speech, Fs, Window_Length, NFFT, No_Filter)
        lfccs = extract_lfcc(speech, Fs, Window_Length, NFFT, No_Filter)

        if mfccs is not None:
            all_mfccs.append(mfccs)
        else:
            print(f"Warning: Sample {idx} has invalid MFCCs.")

        if lfccs is not None:
            all_lfccs.append(lfccs)
        else:
            print(f"Warning: Sample {idx} has invalid LFCCs.")

    return all_mfccs, all_lfccs

def combine_features(mfcc_data, lfcc_data, output_file):
    """Combines MFCC and LFCC features into a single CSV file."""
    combined_data = []

    for mfcc, lfcc in zip(mfcc_data, lfcc_data):
        combined_features = np.hstack((mfcc.flatten(), lfcc.flatten()))
        combined_data.append(combined_features)

    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_file, index=False, header=False)
    print(f"Combined features saved to {output_file}")

if __name__ == "__main__":
    sr = 16000  # Sample rate
    num_samples = 6000  # Number of samples
    samples = [np.random.randn(sr) for _ in range(num_samples)]  # Replace this with actual audio loading

    Window_Length = 25  # Window length in milliseconds
    NFFT = 512  # Number of FFT points
    No_Filter = 20  # Number of Mel filters (MFCC dimensions)

    # Extract features
    all_mfccs, all_lfccs = extract_features(samples, sr, Window_Length, NFFT, No_Filter)

    # Combine and save features
    output_file = r'C:\Users\Serilda\Desktop\Final Year Project\Voice_Spoofing_Detection\combined_features.csv'  # Update with desired output path
    combine_features(all_mfccs, all_lfccs, output_file)
