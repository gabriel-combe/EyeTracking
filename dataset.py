import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset

pd.options.mode.chained_assignment = None  # default='warn'

class EyeDataset(Dataset):
    def __init__(self, csv_files: List[str], root_dir: str, transform=None):
        # Read all csv files
        list_csv = []
        for csv_file in csv_files:
            current_csv = pd.read_csv(os.path.join(root_dir, csv_file), comment='#', index_col=None, header=0)
            for i in range(len(current_csv['imagefile'])):
                current_csv['imagefile'][i] = os.path.join(csv_file.split('.')[0], current_csv.iloc[i, 0])
            list_csv.append(current_csv)

        self.annotations = pd.concat(list_csv, axis=0, ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 2:], dtype=torch.float32)

        if self.annotations.iloc[index, 1] == 'L':
            y_label[0] *= -1.0
            image = np.fliplr(image)

        if self.transform:
            image = self.transform(image.copy())
        
        return (image, y_label)