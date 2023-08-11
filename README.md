# EyeTracking

A simple gaze tracking model using Resnet-18 in pytorch.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

You need to install `pipenv`, a tool for managing Python virtual environments. You can install `pipenv` by running:

```bash
pip install pipenv
```

## Installation
1. Clone the repository to your local machine:
```bash
git clone https://github.com/gabriel-combe/EyeTracking.git
```
2. Navigate to the root directory of the project:
```bash
cd EyeTracking
```
3. Install the project dependencies using pipenv:
```bash
pipenv install
```
This will install all the necessary dependencies specified in the Pipfile and Pipfile.lock.

## Usage
Once the dependencies are installed, you can run the training script by executing:
```bash
pipenv run python train.py --epochs 50 --batch-size 256 --epsilon 0.0001 --skip-epoch 5 --lr 0.001
```

You can also run the inference script (WIP) by executing:
```bash
pipenv run python infer.py --weight 'weights/best.pth'
```