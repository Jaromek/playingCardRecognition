import torch
import torch.optim as optim
import torchvision.transforms as transforms
import neuralNetwork


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
    model = neuralNetwork.ConvNN(num_classes=10).to(EvaluateNN.device)
    criterion = neuralNetwork.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)