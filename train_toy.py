from models import ToyModel
from dataset import ImagesData
import tensorflow as tf
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
import torch
from matplotlib import pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import datasets


if __name__ == '__main__':
    device = 'cuda'
    n_epochs = 30
    warmup = 4
    num_workers = 4
    batch_size = 16
    lenth = 2000
    expname = 'toymnist'

    if os.path.exists('/home/palm/data/coco'):
        src = '/home/palm/data/coco'
    else:
        src = '/media/palm/data/coco/images'
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                                download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                               download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = ToyModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    image = Image.open('/home/palm/Pictures/Turkish-Angora-Cat-compressed-768x384.jpg').convert('RGB')
    losses = []
    steps = 0
    dst = '/media/palm/Data/wher_is_clip/output'
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image, _) in enumerate(train_loader):
            image = image.to(device)
            recon = model(image)
            loss = criterion(recon, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy())]
            progbar.update(idx + 1, printlog)
            steps += 1

        with torch.no_grad():
            for idx, (image, _) in enumerate(train_loader):
                image = image.to(device)
                recon = model(image)
                break
        recon2 = recon.cpu().detach().numpy()
        recon2 = recon2[:, 0]
        recon2 = np.hstack(recon2)
        image2 = image.cpu().detach().numpy()
        image2 = image2[:, 0]
        image2 = np.hstack(image2)
        plt.imsave(
            f'/media/palm/Data/wher_is_clip/output/{expname}/{epoch:02d}.png',
            np.vstack((image2, recon2)),
            cmap='gray'
        )
