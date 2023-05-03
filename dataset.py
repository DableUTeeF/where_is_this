from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImagesData(Dataset):
    def __init__(self, path, size=224):
        self.path = path
        self.data = os.listdir(path)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path, self.data[idx])).convert('RGB')
        image = self.transform(image)
        return image
