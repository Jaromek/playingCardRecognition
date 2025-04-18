import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt


class DataPreparation:
    
    def __init__(self, train_path, val_path, test_path, image_size=(224, 224), batch_size=32):
        self.TRAIN_PATH = train_path
        self.VAL_PATH = val_path
        self.TEST_PATH = test_path
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size

    def prepare_data(self, data_shape=True):

        """
        Function for preparing data to train a neural network.
        Returns DataLoader objects for the training, validation, and test datasets.
        The data is prepared for the PyTorch library.
        The function arguments are initialized in the class constructor, and within the function
        we can only display the shapes of the images and labels under the variable `data_shape`,
        which is set to True by default.

        transform - image preprocessing and augmentation  
        train_dataset - training dataset  
        val_dataset - validation dataset  
        test_dataset - test dataset  
        train_loader - DataLoader object for the training dataset  
        val_loader - DataLoader object for the validation dataset  
        test_loader - DataLoader object for the test dataset  
        """


            
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(root=self.TRAIN_PATH, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=self.VAL_PATH, transform=val_test_transform)
        test_dataset = datasets.ImageFolder(root=self.TEST_PATH, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

        if data_shape:
            for images, labels in train_loader:
                print("Images shape:", images.shape)
                print("Labels shape:", labels.shape)
                break

        return train_loader, val_loader, test_loader

class DataAugmentation:
    def __init__(self, train_loader, val_loader, test_loader, deg):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.deg = deg
    


class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNN, self).__init__()
        self.features = nn.Sequential(
           
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.Dropout(0.4),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(0.5),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EvaluateNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model, train_loader, test_loader, optimizer, criterion):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train(model, train_loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training", leave=True)

        for batch_idx, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                progress_bar.set_postfix(l=running_loss / 100)
                running_loss = 0.0


    def test(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(test_loader, total=len(test_loader), desc="Testing", leave=True)

        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(torch.softmax(outputs, dim=1), 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar.set_postfix(l=test_loss / len(test_loader), accuracy=100. * correct / total)

            accuracy = 100. * correct / total

        print(f"Test loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy}%")

        return accuracy




if __name__ == '__main__':
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    dir_path = 'dataset'
    TRAIN_PATH = f'{dir_path}/train'
    VAL_PATH = f'{dir_path}/valid'
    TEST_PATH = f'{dir_path}/test'

    device = EvaluateNN.device
    print(f"Used device: {device}")

    train_loader, val_loader, test_loader = DataPreparation(TRAIN_PATH, VAL_PATH, TEST_PATH).prepare_data()

    num_classes = len(train_loader.dataset.classes)
    print(f"Class number: {num_classes}")

    model = ConvNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    accuracy = []

    num_epochs = 150
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        EvaluateNN.train(model, train_loader, optimizer, criterion, device)
        acc = EvaluateNN.test(model, test_loader, criterion, device)
        scheduler.step()

        accuracy.append(acc)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, num_epochs + 1), y=accuracy, markers="o", palette="gist_rainbow")
    plt.show()

    