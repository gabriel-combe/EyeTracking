import torch
import numpy as np
import torch.nn as nn
import torch.quantization
from parser import get_opts_inference
from model import EyeTrackingV1, EyeTrackingV2

# For importing data
import torchvision
from dataset import EyeDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Visualization
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=='__main__':
    # Get parameters
    args = get_opts_inference()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Transformations
    transform = T.Compose([ T.ToTensor(), # Normalizes to range [0,1]
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Further normalization
                        ])

    # Loading Dataset
    dataset = EyeDataset(csv_files=['14.csv'], root_dir='dataset', transform=transform)
    inference_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    # Loading Model
    resnet = EyeTrackingV1()

    # Load the saved state_dict into the model
    resnet.load(args.weight)
    resnet.eval()

    if args.quantization:
        # Quantize the entire model
        torch.quantization.quantize_dynamic(
            resnet,
            dtype=torch.qint8,
            inplace=True
        )
    else:
        resnet.to(device)

    for X, y in tqdm(inference_loader, desc=f'Inference test'):
        # Denormalize input image
        img = X[0].numpy().transpose((1,2,0))
        img = img-np.min(img)
        img = img/np.max(img)

        # Push data to GPU
        if not args.quantization:
            X = X.to(device)

        start_time = time.time()

        # Perform inference with the model
        with torch.no_grad():
            output = resnet(X)
        
        elapsed_time = (time.time()-start_time)*1000.0

        plt.imshow(img)
        plt.title(f"\nElapsed time: {elapsed_time:.4f} ms")
        plt.text(0,0,f'{y[0][0]:.4f}, {y[0][1]:.4f}',ha='left',va='top',fontweight='bold',color='k',backgroundcolor='g')
        plt.text(img.shape[1],0,f'{output[0][0]:.4f}, {output[0][1]:.4f}',ha='right',va='top',fontweight='bold',color='k',backgroundcolor='r')
        plt.axis('off')

        plt.show()