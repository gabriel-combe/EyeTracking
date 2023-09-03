import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class EyeTracking_resnet50(nn.Module):
    def __init__(self):
        super(EyeTracking_resnet50, self).__init__()

        # Import ResNet-18 model
        self.resnet = torchvision.models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.resnet.fc = nn.Linear(2048, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor):
        out = self.lrelu(self.resnet(x))
        out = self.lrelu(self.fc1(out))
        out = self.lrelu(self.fc2(out))
        out = self.fc3(out)

        return out
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def train(self):
        self.resnet.train()
    
    def eval(self):
        self.resnet.eval()

class EyeTracking_resnet34(nn.Module):
    def __init__(self):
        super(EyeTracking_resnet34, self).__init__()

        # Import ResNet-18 model
        self.resnet = torchvision.models.resnet34(weights='DEFAULT')

        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.resnet.fc = nn.Linear(512, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor):
        out = self.lrelu(self.resnet(x))
        out = self.lrelu(self.fc1(out))
        out = self.lrelu(self.fc2(out))
        out = self.fc3(out)

        return out
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def train(self):
        self.resnet.train()
    
    def eval(self):
        self.resnet.eval()

class EyeTrackingV2_resnet18(nn.Module):
    def __init__(self):
        super(EyeTrackingV2_resnet18, self).__init__()

        # Import ResNet-18 model
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')

        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.resnet.fc = nn.Linear(512, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(514, 256)
        self.fc3 = nn.Linear(256, 2)

        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor, headpose: torch.Tensor):
        out = self.lrelu(self.resnet(x))
        out = self.lrelu(self.fc1(out))
        out = torch.cat((out, headpose), 1)
        out = self.lrelu(self.fc2(out))
        out = self.fc3(out)

        return out
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def train(self):
        self.resnet.train()
    
    def eval(self):
        self.resnet.eval()
    
class EyeTrackingV1_resnet18(nn.Module):
    def __init__(self):
        super(EyeTrackingV1_resnet18, self).__init__()

        # Import ResNet-18 model
        self.resnet = torchvision.models.resnet18(weights='DEFAULT')

        # Freeze all layers
        for p in self.resnet.parameters():
            p.requires_grad = False

        self.resnet.fc = nn.Linear(512, 1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.lrelu = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor):
        out = self.lrelu(self.resnet(x))
        out = self.lrelu(self.fc1(out))
        out = self.lrelu(self.fc2(out))
        out = self.fc3(out)

        return out
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def train(self):
        self.resnet.train()
    
    def eval(self):
        self.resnet.eval()