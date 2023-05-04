from models import WhereIsCLIP
import torch
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

if __name__ == '__main__':
    model = WhereIsCLIP()
    checkpoint = torch.load('cp/std_1/best.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open('/home/palm/Pictures/Turkish-Angora-Cat-compressed-768x384.jpg').convert('RGB')
    image = transform(image)

    with torch.no_grad():
        recon = model(image.unsqueeze(0))
    recon2 = recon.cpu().detach().numpy()
    plt.imshow(recon2[0, 0])
    plt.figure()
    plt.imshow(image[0])
