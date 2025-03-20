import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataPreparation:
    
    def __init__(self, train_path, val_path, test_path, image_size=(224, 224), batch_size=32):
        self.TRAIN_PATH = train_path
        self.VAL_PATH = val_path
        self.TEST_PATH = test_path
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size

    def prepare_data(self, data_shape=True):

        """
        Funkcja przygotowująca dane do trenowania sieci neuronowej.
        Zwraca obiekty DataLoader dla zbiorów treningowego, walidacyjnego i testowego.
        Dane są przygotowywane pod bibliotekę PyTorch. 
        Argumenty funkcji są inicjalizowane w konstruktorze klasy, a w samej funkcji możemy
        jedyne co to wyświetlić kształty obrazów i etykiet pod zmienną data_shape, która domyślnie
        jest ustawiona na True.
        """

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

        if data_shape:
            for images, labels in train_loader:
                print("Kształt obrazów:", images.shape)
                print("Kształt etykiet:", labels.shape)
                break

        return train_loader, val_loader, test_loader

class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def train(model, train_loader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (batch_idx+1) % 100 == 0:
                print(f"Batch {batch_idx+1}, Strata: {running_loss/100:.4f}")
                running_loss = 0.0

    def test(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(f"Testowa strata: {test_loss/len(test_loader):.4f}, Dokładność: {100.*correct/total:.2f}%")

    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     print(f"Epoka: {epoch+1}/{num_epochs}")
    #     train(model, train_loader, optimizer, criterion, device)
    #     test(model, test_loader, criterion, device)

if __name__ == '__main__':
    print(f"Używane urządzenie: {EvaluateNN.device}")
    model = ConvNN(num_classes=10).to(EvaluateNN.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)