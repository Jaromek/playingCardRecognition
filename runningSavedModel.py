from NeuralNetworkMain import DataLoader, EvaluateNN, ConvNN, torch, datasets, transforms, nn, plt, sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

if __name__ == '__main__':
    MODEL_PATH = 'acc81.5/best_model.pth'
    DIR_PATH = 'dataset'
    TRAIN_PATH = f'{DIR_PATH}/train'
    BATCH_SIZE = 32

    criterion = nn.CrossEntropyLoss()
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=val_test_transform)

    model = ConvNN(num_classes=53)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_splits = 5
    subset_size = len(test_dataset) // num_splits
    dokladnosc = []

    all_preds = []
    all_targets = []

    for i in range(num_splits):
        indices = list(range(i * subset_size, (i + 1) * subset_size))
        subset = torch.utils.data.Subset(test_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        dokl = EvaluateNN.test(model, subset_loader, criterion, device)
        dokladnosc.append(dokl)

        with torch.no_grad():
            for images, labels in subset_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

    def map_to_color(label: list) -> list:
        color_map = ("red", "black")

    def map_to_suits(label: list) -> list:
        suits_map = ("spades", "hearts", "diamonds", "clubs")

    def map_to_ranks(label: list) -> list:
        ranks_map = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
        return [ranks_map[l] for l in label]


    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, num_splits + 1), y=dokladnosc, markers="o", palette="gist_rainbow")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
