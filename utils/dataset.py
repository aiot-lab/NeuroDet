#############################################
# Author: Luca Yu
# Date: 2024-08-12
# Description: Dataset class
#############################################

import torch
from loguru import logger
import os
import pickle
from torch.utils.data import Dataset

class lssDataset(Dataset):
    def __init__(self, phase, dataset_paths, calibration_paths):
        assert len(dataset_paths) == len(calibration_paths), "Each dataset path must have a corresponding calibration path"
        
        self.phase = phase
        self.data_list = []
        self.dataset_paths = dataset_paths
        self.calibration_paths = calibration_paths
        self.calibration_spectrums = []

        # Load data paths
        for dataset_path in self.dataset_paths:
            data_path = os.path.join(dataset_path, phase)
            self.data_list += [os.path.join(data_path, data) for data in os.listdir(data_path)]

        self.calibration = True
        if self.calibration:
            logger.info('Calibration mode enabled')
            for calibration_path in self.calibration_paths:
                calibration_files = os.listdir(calibration_path)
                calibration_spectrum = torch.zeros(87, 128)
                for calibration_file in calibration_files:
                    with open(os.path.join(calibration_path, calibration_file), 'rb') as f:
                        calibration_data = pickle.load(f)
                    calibration_data = calibration_data['spectrum']
                    calibration_data = torch.tensor(calibration_data, dtype=torch.float32).flip(0)
                    calibration_spectrum += torch.tensor(calibration_data, dtype=torch.float32)
                calibration_spectrum = calibration_spectrum / len(calibration_files)
                self.calibration_spectrums.append(calibration_spectrum)  # Correctly place this line inside the loop

    def __len__(self):
        logger.info(f'Dataset Loaded, Size: {len(self.data_list)}')
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx]

        # Determine which dataset and corresponding calibration spectrum to use
        for i, dataset_path in enumerate(self.dataset_paths):
            if data_path.startswith(dataset_path):
                calibration_spectrum = self.calibration_spectrums[i]
                break
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        spectrum = torch.tensor(data['spectrum'], dtype=torch.float32)
        pointcloud = torch.tensor(data['pointcloud'], dtype=torch.float32)

        if self.calibration:
            spectrum = spectrum - calibration_spectrum

            pointcloud = torch.where(pointcloud == 1, torch.tensor(0), pointcloud)
            pointcloud = torch.where(pointcloud == 2, torch.tensor(1), pointcloud)

        # Flatten the spectrum and pointcloud
        spectrum = spectrum.flatten()
        pointcloud = pointcloud.flatten()

        result = {
            'pointcloud': pointcloud,
            'spectrum': spectrum
        }
        return result
    
    def _collate_fn(self, batch):
        pointclouds = torch.stack([item['pointcloud'] for item in batch], dim=1)
        spectrums = torch.stack([item['spectrum'] for item in batch], dim=1)
        return {'pointcloud': pointclouds, 'spectrum': spectrums}

