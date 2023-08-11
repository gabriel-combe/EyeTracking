# Import libraries
import numpy as np
from parser import get_opts_train

# Import utils
import os
import wandb
from utils import example_images, setup_logger

import torch
import torch.nn as nn
from model import EyeTrackingV1, EyeTrackingV2
import torch.nn.functional as F

# For importing data
import torchvision
from dataset import EyeDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

# Visualization
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

if __name__=="__main__":
    # Get parameters
    args = get_opts_train()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Setup logger
    run = setup_logger(args)

    # Transformations
    transform = T.Compose([ T.ToTensor(), # Normalizes to range [0,1]
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Further normalization
                        ])

    # Loading Dataset
    dataset = EyeDataset(csv_files=['01.csv', '02.csv', '03.csv', '04.csv', '05.csv', '06.csv', '07.csv', '08.csv', '09.csv', '10.csv', '11.csv', '12.csv', '13.csv', '14.csv'], root_dir='dataset', transform=transform)
    # dataset = EyeDataset(csv_files=['01.csv'], root_dir='dataset', transform=transform)
    train_set, test_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    example_images(train_loader)

    # Loading Model
    resnet = EyeTrackingV1()

    # ResNet18 summary
    # summary(resnet.to(device), [(3, 640, 480), (2,)])

    resnet.to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr, betas=(0.9, 0.999))

    epochs = args.epochs
    epsilon = args.epsilon

    # Initialize losses
    # trainLoss = torch.zeros(epochs)
    # testLoss  = torch.zeros(epochs)
    # trainAcc  = torch.zeros(epochs)
    # testAcc   = torch.zeros(epochs)

    # Create folders
    if not os.path.exists('runs/train'):
        os.makedirs('runs/train')

    # Log gradients
    wandb.watch(resnet, log_freq=1)

    # Loop over all epochs
    for epoch in range(epochs):
        resnet.train() # switch to train mode
        batchLoss = []
        batchAcc  = []

        for X, y in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
            # Push data to GPU
            X = X.to(device)
            y = y.to(device)

            # Forward pass and loss
            pred = resnet(X)
            loss = criterion(pred.float(), y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss and accuracy from this batch
            batchLoss.append(loss.item())
            batchAcc.append(torch.mean(((pred - y) < epsilon).float()).item())

        # Get average losses and accuracies across the batches
        # trainLoss[epoch] = np.mean(batchLoss)
        # trainAcc[epoch] = 100*np.mean(batchAcc)

        # Print out status update for train
        wandb.log({"Train loss": np.mean(batchLoss), "Train accuracy": 100*np.mean(batchAcc)})
        print(f'Train loss = {np.mean(batchLoss):.4f}, Train accuracy = {100*np.mean(batchAcc):.2f}%')

        if epoch%args.skip_epoch == 0:
            # Test performance
            resnet.eval() # Switch to test mode
            batchAcc = []
            batchLoss = []

            for X, y in tqdm(test_loader, desc=f'Test'):
                # Push data to GPU
                X = X.to(device)
                y = y.to(device)

                # Forward pass and loss
                with torch.no_grad():
                    pred = resnet(X)
                    loss = criterion(pred.float(), y.float())

                # Loss and accuracy from this batch
                batchLoss.append(loss.item())
                batchAcc.append(torch.mean(((pred - y) < epsilon).float()).item())
            
            # testLoss[epoch] = np.mean(batchLoss)
            # testAcc[epoch] = 100*np.mean(batchAcc)

            # Print out status update for test
            wandb.log({"Test loss": np.mean(batchLoss), "Test accuracy": 100*np.mean(batchAcc)})
            print(f'Test accuracy = {100*np.mean(batchAcc):.2f}%')

            # Save model's weights
            resnet.save(f'runs/train/epoch_{epoch}.pth')

            # # Log examples

            # X, y = next(iter(test_loader))

            # img = X[:5].cpu().numpy().transpose((0,2,3,1))
            # gt = y[:5].cpu()
            # print(img.shape)
            # gt_gaze_x = gt[:, 0]
            # gt_gaze_y = gt[:, 1]
            # print(gt_gaze_x.size())
            # for i in range(5):
            #     img[i] = img[i]-np.min(img[i])
            #     img[i] = img[i]/np.max(img[i])
            # print(img.shape)

            # # Push data to GPU
            # X = X[:5].to(device)

            # # Forward pass and loss
            # with torch.no_grad():
            #     pred = resnet(X).cpu().numpy()
            #     pred_gaze_x = pred[:, 0]
            #     pred_gaze_y = pred[:, 1]
            
            # print(gt_gaze_x.numpy())
            # print(gt_gaze_x.numpy().tolist())

            # table_eyetracking = wandb.Table(
            #     columns=["gt_gaze_x", "gt_gaze_y", "image", "predicted_gaze_x", "predicted_gaze_y"],
            #     data=[gt_gaze_x.numpy()[0], gt_gaze_x.numpy()[0], img[0], pred_gaze_x[0], pred_gaze_y[0]]
            # )

            # wandb.log({"EyeTracking_predictions": table_eyetracking})
                


    
    # Visualizing performance
    # fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # ax[0].plot(trainLoss, 's-', label='Train')
    # ax[0].plot(testLoss, 'o-', label='Test')
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('Loss (MSE)')
    # ax[0].set_title('Model loss')
    # ax[0].legend()

    # ax[1].plot(trainAcc, 's-', label='Train')
    # ax[1].plot(testAcc, 'o-', label='Test')
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('Accuracy (%)')
    # ax[1].set_title(f'Final model train/test accuracy: {trainAcc[-1]:.2f}/{testAcc[-1]:.2f}%')
    # ax[1].legend()

    # plt.suptitle('Pretrained ResNet-18 (+ 3 fc layers) on NVGaze dataset', fontweight='bold', fontsize=14)
    # plt.show()
