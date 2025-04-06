from PIL import Image
import torchvision.transforms.functional as F
from NeuralNetworkMain import DataLoader, EvaluateNN, ConvNN, torch, datasets, transforms, nn, plt, sns
import torch
import os 

MODEL_PATH = 'acc81.5/best_model.pth'
DIR_PATH = 'dataset'
TRAIN_PATH = f'{DIR_PATH}/train'
BATCH_SIZE = 32

val_test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

test_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=val_test_transform)

from PIL import Image
import torchvision.transforms.functional as F
import os

# Ścieżka do przykładowego obrazka
sample_image_path = f'{TRAIN_PATH}/{test_dataset.classes[2]}/{os.listdir(TRAIN_PATH + "/" + test_dataset.classes[2])[5]}'

# Oryginalny obrazek (PIL)
original_img = Image.open(sample_image_path).convert("RGB")

# Po transformacji (Tensor z normalizacją)
transformed_img = val_test_transform(original_img)

# Odwracanie normalizacji (nowy tensor)
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

denorm_img = denormalize(transformed_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
denorm_img = torch.clamp(denorm_img, 0, 1)  # upewniamy się że wartości są w [0,1]

# Wyświetlenie obrazków
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(denorm_img.permute(1, 2, 0).numpy())
plt.axis("off")

plt.show()
