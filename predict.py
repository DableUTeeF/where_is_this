from models import SimPlerModel
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import cv2
plt.axis('off')


if __name__ == '__main__':
    model = SimPlerModel()
    checkpoint = torch.load('cp/mnet_1/best.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open('/media/palm/data/ppa/cocoim/000000000872.jpg').convert('RGB')
    for i in range(20):
        image2 = transform(image)

        with torch.no_grad():
            recon = model(image2.unsqueeze(0))
        recon2 = recon.cpu().detach()
        plt.figure()
        plt.imshow(np.hstack((image2.permute(1, 2, 0), cv2.resize(recon2[0].permute(1, 2, 0).numpy(), (224, 224)))))
    plt.show()
