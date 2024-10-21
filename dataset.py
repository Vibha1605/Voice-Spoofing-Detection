import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ASVspoof2021(Dataset):
    def __init__(self, path_to_features, path_to_protocol, part='train'):
        self.path_to_features = path_to_features
        self.part = part
        self.path_to_protocol = path_to_protocol
        
        # Load protocol
        protocol_file = os.path.join(self.path_to_protocol, f'ASVspoof2021.LA.cm.{self.part}.trl.txt')
        with open(protocol_file, 'r') as f:
            self.audio_info = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.audio_info)

    def __getitem__(self, idx):
        # Extract filename
        filename = self.audio_info[idx][1]
        
        # Load features from CSV
        lfcc_csv = os.path.join(self.path_to_features, f'{filename}_LFCC.csv')
        mfcc_csv = os.path.join(self.path_to_features, f'{filename}_MFCC.csv')
        
        # Load LFCC and MFCC features from CSV files
        lfcc_feat = pd.read_csv(lfcc_csv, header=None).values
        mfcc_feat = pd.read_csv(mfcc_csv, header=None).values
        print(f"Looking for file: {lfcc_csv}")  # To debug LFCC file path

        # Convert to tensors
        lfcc_feat = torch.tensor(lfcc_feat, dtype=torch.float32)
        mfcc_feat = torch.tensor(mfcc_feat, dtype=torch.float32)
        
        # Get the label and tag from protocol
        tag = self.audio_info[idx][3]
        label = self.audio_info[idx][4]
        
        # Convert labels to int (e.g., bonafide = 0, spoof = 1)
        tag_mapping = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                       "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                       "A19": 19}
        label_mapping = {"spoof": 1, "bonafide": 0}
        
        return lfcc_feat, mfcc_feat, filename, tag_mapping[tag], label_mapping[label]

    def collate_fn(self, samples):
        lfcc_feats = [s[0] for s in samples]
        mfcc_feats = [s[1] for s in samples]
        filenames = [s[2] for s in samples]
        tags = [s[3] for s in samples]
        labels = [s[4] for s in samples]
        return lfcc_feats, mfcc_feats, filenames, tags, labels