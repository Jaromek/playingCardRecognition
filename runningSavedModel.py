from NeuralNetworkMain import EvaluateNN, torch, DataLoader, datasets, transforms, nn

MODEL_PATH = 'acc81.5/best_model.pth'
DIR_PATH = 'I:/playingCards'
TRAIN_PATH = f'{DIR_PATH}/train'
BATCH_SIZE = 32
model = torch.load(MODEL_PATH)
criterion = nn.CrossEntropyLoss()
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EvaluateNN.test(model, test_loader, criterion, device)