from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataPreparation:
    
    def __init__(self, train_path, val_path, test_path, image_size=(224, 224), batch_size=32):
        self.TRAIN_PATH = train_path
        self.VAL_PATH = val_path
        self.TEST_PATH = test_path
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size

    def prepare_data(self):

        transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = datasets.ImageFolder(root=self.TRAIN_PATH, transform=transform)
        val_dataset = datasets.ImageFolder(root=self.VAL_PATH, transform=transform)
        test_dataset = datasets.ImageFolder(root=self.TEST_PATH, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

        for images, labels in train_loader:
            print("Kształt obrazów:", images.shape)
            print("Kształt etykiet:", labels.shape)
            break
