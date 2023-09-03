import argparse

def get_opts_train():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size for the model')
    parser.add_argument('--skip-epoch', type=int, default=4, help='Number of epoch to skip between test display')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the model')
    parser.add_argument('--epsilon', type=float, default=0.001, help='Error allowed in the prediction')

    # Seed for random generator
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random generator')


    return parser.parse_args()

def get_opts_inference():
    parser = argparse.ArgumentParser()
    
    # Inference args
    parser.add_argument('--quantization', action='store_true', help='Quantize or not the model')
    parser.add_argument('--weight', type=str, default='weights/best.pth', help='Weight of the model')
    parser.add_argument('--epsilon', type=float, default=0.001, help='Error allowed in the prediction')
    parser.add_argument('--image', type=str, default=None, help='Image to use for the inference')

    # Seed for random generator
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random generator')


    return parser.parse_args()