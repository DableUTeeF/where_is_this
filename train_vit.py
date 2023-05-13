import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from transformers import AutoModel, CLIPProcessor
from models import WhereIsFeatures
from dataset import FolderData
import tensorflow as tf
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import datasets


def clip_accuracy(text_embeds, image_embeds, labels):
    logits_per_image = torch.matmul(text_embeds, image_embeds.t()).t()
    probs = logits_per_image.softmax(dim=1)
    return (probs.argmax(1) == labels).float().mean()


def clip_tanh_accuracy(text_embeds, image_embeds, labels):
    logits_per_image = torch.matmul(text_embeds*2-1, (image_embeds*2-1).t()).t()
    probs = logits_per_image.softmax(dim=1)
    return (probs.argmax(1) == labels).float().mean()


def xor_acc(text_embeds, image_embeds, labels):
    logits_per_image = torch.matmul(text_embeds, image_embeds.t()).t()
    probs = logits_per_image.softmax(dim=1)
    return (probs.argmax(1) == labels).float().mean()


if __name__ == '__main__':
    device = 'cuda'
    n_epochs = 5
    warmup = 4
    num_workers = 4
    batch_size = 16
    lenth = 2000
    expname = 'multistage_smalldogcat'

    if os.path.exists('/home/palm/data/dogs-vs-cats'):
        src = '/home/palm/data/dogs-vs-cats'
    else:
        src = '/media/palm/data/Dogs_vs_Cats'

    classes = ['cat', 'dog']
    train_dataset = FolderData(f'{src}/train', size=224)
    val_dataset = FolderData(f'{src}/val', size=224)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], return_tensors="pt", padding=True)

    mse = nn.MSELoss()
    sigmoid = nn.Sigmoid()

    clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
    for param in clip.parameters():
        param.requires_grad = False
    vision_model = clip.vision_model
    visual_projection = clip.visual_projection
    text_projection = clip.text_projection
    prompts = clip.text_model(**inputs.to('cuda'))
    prompts = sigmoid(text_projection(prompts[1]))
    model = WhereIsFeatures()
    model = model.to(device)

    # autoencoder: encoder/decoder
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=5,
                                 t_mul=1,
                                 lr_min=5e-5,
                                 decay_rate=0.1,
                                 cycle_limit=1,
                                 t_in_epochs=False,
                                 noise_range_t=None,
                                 )
    for epoch in range(1):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image, _, cls) in enumerate(train_loader):
            image = image.to(device)
            cls = cls.to(device)
            with torch.no_grad():
                features = vision_model(image)['pooler_output']
                features = visual_projection(features)
                features = sigmoid(features)
                std_acc = clip_accuracy(prompts, features, cls)

            x = model.encode(features)
            recon = model.decode(x)
            recon_acc = clip_accuracy(prompts, recon, cls)
            loss = mse(recon, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy()),
                        ('std_acc', std_acc.cpu().detach().numpy()),
                        ('recon_acc', recon_acc.cpu().detach().numpy()),
                        ]
            progbar.update(idx + 1, printlog)

    # autoencoder: buffer nowhere
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=n_epochs,
                                 t_mul=1,
                                 lr_min=5e-5,
                                 decay_rate=0.1,
                                 cycle_limit=1,
                                 t_in_epochs=False,
                                 noise_range_t=None,
                                 )
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image, _, cls) in enumerate(train_loader):
            image = image.to(device)
            cls = cls.to(device)
            with torch.no_grad():
                features = vision_model(image)['pooler_output']
                features = visual_projection(features)
                features = sigmoid(features)
                std_acc = clip_accuracy(prompts, features, cls)
                prompts_ecd = model.encode(prompts)
                _, prompts_ecd, _ = model.where(prompts_ecd, False)
                x = model.encode(features)
            x, ecd, gt = model.where(x, False)
            buffer_acc = clip_accuracy(prompts_ecd[:, 0], ecd[:, 0], cls)
            buffer_acc2 = clip_tanh_accuracy(prompts_ecd[:, 0], ecd[:, 0], cls)
            loss = mse(x, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy()),
                        ('std_acc', std_acc.cpu().detach().numpy()),
                        ('tanh_acc', buffer_acc2.cpu().detach().numpy()),
                        ('sigmoid_acc', buffer_acc.cpu().detach().numpy()),
                        ]
            progbar.update(idx + 1, printlog)
        if progbar._values['tanh_acc'][0] / progbar._values['std_acc'][0] > 0.95:
            break

    # autoencoder: buffer where
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = CosineLRScheduler(optimizer,
                                 t_initial=n_epochs,
                                 t_mul=1,
                                 lr_min=5e-5,
                                 decay_rate=0.1,
                                 cycle_limit=1,
                                 t_in_epochs=False,
                                 noise_range_t=None,
                                 )
    for epoch in range(n_epochs):
        print('Epoch:', epoch + 1)
        model.train()
        progbar = tf.keras.utils.Progbar(len(train_loader))
        for idx, (image, _, cls) in enumerate(train_loader):
            image = image.to(device)
            cls = cls.to(device)
            with torch.no_grad():
                features = vision_model(image)['pooler_output']
                features = visual_projection(features)
                features = sigmoid(features)
                std_acc = clip_accuracy(prompts, features, cls)
                prompts_ecd = model.encode(prompts)
                _, prompts_ecd, _ = model.where(prompts_ecd, True)
                x = model.encode(features)
            x, ecd, gt = model.where(x, True)
            buffer_acc = clip_accuracy(prompts_ecd[:, 0], ecd[:, 0], cls)
            buffer_acc2 = clip_tanh_accuracy(prompts_ecd[:, 0], ecd[:, 0], cls)
            loss = mse(x, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            printlog = [('loss', loss.cpu().detach().numpy()),
                        ('std_acc', std_acc.cpu().detach().numpy()),
                        ('tanh_acc', buffer_acc2.cpu().detach().numpy()),
                        ('sigmoid_acc', buffer_acc.cpu().detach().numpy()),
                        ]
            progbar.update(idx + 1, printlog)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               f'cp/where_1024/last.pth')

