import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class PumpDataset(Dataset):
    def __init__(self, time_series, scalograms, labels):
        self.time_series = torch.FloatTensor(time_series)
        self.scalograms = torch.FloatTensor(scalograms).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'time_series': self.time_series[idx],
            'scalogram': self.scalograms[idx],
            'label': self.labels[idx]
        }

def load_and_preprocess_data(config):
    # Load the data from the AE_ALL.mat file
    file_path = r'E:\3 Paper MCT\data\AE_ALL.mat'
    data = loadmat(file_path)

    # Extract the non-'I' cases (BF, GF, TF, N) and use only channel 1
    data_dict = {
        'BF': data['BF'][0],  # 4x1 cell, take only channel 1 (first element)
        'GF': data['GF'][0],
        'TF': data['TF'][0],
        'N': data['N'][0]
    }

    time_series = []
    labels = []

    # Iterate through the fault types
    for i, (fault_type, data_list) in enumerate(data_dict.items()):
        for j in range(len(data_list)):  # Assuming each fault type has 4 cells
            channel_1_data = data_list[j][:, 0]  # Extract channel 1 data
            time_series.append(channel_1_data)
            labels.append(i)  # Assign a label based on the fault type (BF:0, GF:1, etc.)

    time_series = np.vstack(time_series)
    labels = np.array(labels)

    # Normalize time series data
    scaler = StandardScaler()
    time_series = scaler.fit_transform(time_series)

    # Generate scalograms (you can use your preferred wavelet and scales here)
    scalograms = np.array([
        generate_scalogram(signal, scales=config.SCALES, wavelet=config.WAVELET)
        for signal in time_series
    ])
    
    return time_series, scalograms, labels

def generate_scalogram(signal, scales, wavelet):
    # Generate a continuous wavelet transform scalogram
    coefficients, _ = pywt.cwt(signal, np.arange(1, scales+1), wavelet)
    return np.abs(coefficients)
