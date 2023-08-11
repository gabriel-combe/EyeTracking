import wandb
import numpy as np
import matplotlib.pyplot as plt

def example_images(train_loader):
    X, y = next(iter(train_loader))

    # Inspect images
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for (i, ax) in enumerate(axs.flatten()):
        img = X.data[i].numpy().transpose((1,2,0))
        img = img-np.min(img)
        img = img/np.max(img)

        ax.imshow(img)
        ax.text(0,0,f'{y[i][0]:.4f}, {y[i][1]:.4f}',ha='left',va='top',fontweight='bold',color='k',backgroundcolor='y')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def setup_logger(parser):
    wandb.login()

    run = wandb.init(
        project="EyeTracking",
        config={
            "learning_rate": parser.lr,
            "epochs": parser.epochs,
            "batch_size": parser.batch_size,
            "architecture": "resnet18 + 3FC"
        }
    )

    return run