import warnings
warnings.filterwarnings('ignore')
from models import BigToyModel
from dataset import FolderData
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
    n_epochs = 15
    warmup = 4
    num_workers = 4
    batch_size = 16
    lenth = 2000
    expname = 'where_is_smalldogcat_pretrain'

    if os.path.exists('/home/palm/data/Dogs_vs_Cats'):
        src = '/home/palm/data/Dogs_vs_Cats'
    else:
        src = '/media/palm/data/Dogs_vs_Cats'
    train_dataset = FolderData(f'{src}/train', size=32)
    val_dataset = FolderData(f'{src}/val', size=32)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = BigToyModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=n_epochs,
                                 t_mul=1,
                                 lr_min=1e-5,
                                 decay_rate=0.1,
                                 cycle_limit=1,
                                 t_in_epochs=False,
                                 noise_range_t=None,
                                 )

    mse = nn.MSELoss()
    crossentropy = nn.CrossEntropyLoss()

    image = Image.open('/home/palm/Pictures/Turkish-Angora-Cat-compressed-768x384.jpg').convert('RGB')
    losses = []
    steps = 0
    dst = '/media/palm/Data/wher_is_clip/output'
    # for epoch in range(n_epochs):
    #     print('Epoch:', epoch + 1)
    #     model.train()
    #     progbar = tf.keras.utils.Progbar(len(train_loader))
    #     for idx, (image, _, labels) in enumerate(train_loader):
    #         image = image.to(device)
    #         labels = labels.cuda()
    #         recon = model(image, True)
    #         loss = crossentropy(recon, labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         printlog = [('loss', loss.cpu().detach().numpy())]
    #         progbar.update(idx + 1, printlog)
    #         steps += 1
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        hm = 'whereis' if epoch > 3 else 'isnowhere'
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image, _, _) in enumerate(train_loader):
            image = image.to(device)
            recon = model(image, False, epoch > 3)
            loss = mse(recon, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy())]
            progbar.update(idx + 1, printlog)
            steps += 1

        with torch.no_grad():
            for idx, (image, _, _) in enumerate(train_loader):
                image = image.to(device)
                recon = model(image, False, epoch > 3)
                break
        recon2 = recon.cpu().detach().permute(0, 2, 3, 1).numpy()
        recon2 = recon2[:, :]
        recon2 = np.hstack(recon2)
        image2 = image.cpu().detach().permute(0, 2, 3, 1).numpy()
        image2 = image2[:, :]
        image2 = np.hstack(image2)
        plt.imsave(
            f'/media/palm/Data/wher_is_clip/output/{expname}/{epoch:02d}_{hm}.png',
            np.vstack((image2, recon2)),
            cmap='gray'
        )
        schedule.step(epoch+1)
