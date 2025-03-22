from NeuralNetworkMain import DataLoader, EvaluateNN, ConvNN, torch, datasets, transforms, nn, plt, sns


if __name__ == '__main__':
    MODEL_PATH = 'acc81.5/best_model.pth'
    DIR_PATH = 'I:/playingCards'
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

    for i in range(num_splits):
        indices = list(range(i * subset_size, (i + 1) * subset_size))
        subset = torch.utils.data.Subset(test_dataset, indices)
        subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        dokl = EvaluateNN.test(model, subset_loader, criterion, device)
        dokladnosc.append(dokl)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, num_splits + 1), y=dokladnosc, markers="o", palette="gist_rainbow")
    plt.xlabel("Podzbiór testowy")
    plt.ylabel("Dokładność")
    plt.title("Dokładność modelu na różnych fragmentach zbioru testowego")
    plt.show()