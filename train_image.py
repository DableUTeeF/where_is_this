from models import WhereIsCLIP
from dataset import ImagesData
import tensorflow as tf
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
import torch
import numpy as np
import os

if __name__ == '__main__':
    device = 'cuda'
    n_epochs = 30
    num_workers = 4
    batch_size = 16

    if os.path.exists('/home/palm/data/coco'):
        src = '/home/palm/data/coco'
    else:
        src = '/media/palm/data/coco/images'

    train_dataset = ImagesData(f'{src}/train2017')
    val_dataset = ImagesData(f'{src}/val2017')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            )

    model = WhereIsCLIP()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.4)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=n_epochs*len(train_loader),
                                 t_mul=1,
                                 lr_min=1e-8,
                                 decay_rate=0.1,
                                 warmup_t=int(n_epochs*0.1)*len(train_loader),
                                 warmup_lr_init=1e-6,
                                 cycle_limit=1,
                                 t_in_epochs=False,
                                 noise_range_t=None,

                                 )
    min_loss = float('inf')
    criterion = nn.MSELoss()
    steps = 0
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image) in enumerate(train_loader):
            image = image.to(device)
            recon = model(image)
            loss = criterion(recon, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy())]
            progbar.update(idx + 1, printlog)
            schedule.step(steps + 1)
            steps += 1
        model.eval()
        progbar = tf.keras.utils.Progbar(len(val_loader))
        all_losses = []
        with torch.no_grad():
            for idx, (image) in enumerate(val_loader):
                image = image.to(device)
                recon = model(image)
                loss = criterion(recon, image)
                all_losses.append(loss.cpu().detach().numpy())
                printlog = [('loss', loss.cpu().detach().numpy()), ]
                progbar.update(idx + 1, printlog)
        all_losses = float(np.mean(all_losses))
        if all_losses < min_loss:
            min_loss = all_losses
            torch.save({'model': model.state_dict(),
                        'loss': all_losses / len(val_loader),
                        'optimizer': optimizer.state_dict()},
                       f'cp/std_1/best.pth')

        torch.save({'model': model.state_dict(),
                    'loss': all_losses / len(val_loader),
                    'optimizer': optimizer.state_dict()},
                   f'cp/std_1/last.pth')
