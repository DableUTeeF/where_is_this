from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import json


class COCOCaptionData(Dataset):
    def __init__(self, json_path):
        data = json.load(open(json_path))
        self.data = data['annotations']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        return ann['caption'], ann['image_id']


class FolderData(Dataset):
    def __init__(self, path, size=224, mul=1.5):
        self.path = path
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.input_resize = transforms.Compose([
            transforms.Resize(int(size*mul)),
            transforms.RandomCrop((size, size)),
            transforms.ToTensor(),
        ])
        self.output_resize = transforms.Resize(size // 8)
        self.data = []
        for idx, folder in enumerate(sorted(os.listdir(path))):
            for file in os.listdir(os.path.join(path, folder)):
                self.data.append((folder, file, idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder, file, cls = self.data[idx]
        image = Image.open(os.path.join(self.path, folder, file)).convert('RGB')
        image = self.input_resize(image)
        # image = self.transform(image)
        return image, self.output_resize(image), cls

class FolderNameData(FolderData):
    def __getitem__(self, idx):
        folder, file, cls = self.data[idx]
        image = Image.open(os.path.join(self.path, folder, file)).convert('RGB')
        image = self.input_resize(image)
        # image = self.transform(image)
        return image, self.output_resize(image), folder


class ImagesData(Dataset):
    def __init__(self, path, size=224):
        self.path = path
        self.data = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.input_resize = transforms.Compose([
            transforms.Resize(int(size*1.5)),
            transforms.RandomCrop((size, size)),
            transforms.ToTensor(),
        ])
        self.output_resize = transforms.Resize(size // 8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path, self.data[idx])).convert('RGB')
        image = self.input_resize(image)
        return self.transform(image), self.output_resize(image)


class RandomData(Dataset):
    def __init__(self, length, size=224):
        self.length = length
        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.transform(torch.rand((3, 224, 224), dtype=torch.float))
