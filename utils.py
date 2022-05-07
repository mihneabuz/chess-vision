import numpy as np
import torch
import torchsummary
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

cells = {
    "A1": [0, 0], "A2": [0, 1], "A3": [0, 2], "A4": [0, 3], "A5": [0, 4], "A6": [0, 5], "A7": [0, 6], "A8": [0, 7],
    "B1": [1, 0], "B2": [1, 1], "B3": [1, 2], "B4": [1, 3], "B5": [1, 4], "B6": [1, 5], "B7": [1, 6], "B8": [1, 7],
    "C1": [2, 0], "C2": [2, 1], "C3": [2, 2], "C4": [2, 3], "C5": [2, 4], "C6": [2, 5], "C7": [2, 6], "C8": [2, 7],
    "D1": [3, 0], "D2": [3, 1], "D3": [3, 2], "D4": [3, 3], "D5": [3, 4], "D6": [3, 5], "D7": [3, 6], "D8": [3, 7],
    "E1": [4, 0], "E2": [4, 1], "E3": [4, 2], "E4": [4, 3], "E5": [4, 4], "E6": [4, 5], "E7": [4, 6], "E8": [4, 7],
    "F1": [5, 0], "F2": [5, 1], "F3": [5, 2], "F4": [5, 3], "F5": [5, 4], "F6": [5, 5], "F7": [5, 6], "F8": [5, 7],
    "G1": [6, 0], "G2": [6, 1], "G3": [6, 2], "G4": [6, 3], "G5": [6, 4], "G6": [6, 5], "G7": [6, 6], "G8": [6, 7],
    "H1": [7, 0], "H2": [7, 1], "H3": [7, 2], "H4": [7, 3], "H5": [7, 4], "H6": [7, 5], "H7": [7, 6], "H8": [7, 7]
}

classes_dict = {
    'empty': 0,
    'pawn_w': 1,
    'pawn_b': 2,
    'bishop_w': 3,
    'bishop_b': 4,
    'knight_w': 5,
    'knight_b': 6,
    'rook_w': 7,
    'rook_b': 8,
    'queen_w': 9,
    'queen_b': 10,
    'king_w': 11,
    'king_b': 12,
}

num_classes = len(classes_dict)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def summary(model, input):
    torchsummary.summary(model, input_size=input, device='cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, dataloader, optimizer, criterion, transform=lambda x: x):
    device = get_device()
    model.train(True)

    losses = []
    for images, labels in tqdm(dataloader, desc='batches'):
        images = transform(images.to(device))
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return losses

def validation_metrics(model, dataloader, transform, results):
    device = get_device()
    model.train(False)

    with torch.no_grad():
        preds = np.array([])
        real = np.array([])

        for images, labels in tqdm(dataloader, desc='metrics'):
            images = transform(images.to(device))
            labels = labels.to(device)

            preds = np.concatenate((preds, results(model(images)).cpu().numpy()))
            real = np.concatenate((real, labels.cpu().numpy()))

        accuracy = accuracy_score(real, preds),
        f1 = f1_score(real, preds, labels=range(num_classes), average='macro'),

        precision = precision_score(real, preds, labels=range(num_classes), average=None),
        recall = recall_score(real, preds, labels=range(num_classes), average=None)

    return accuracy[0], f1[0], precision[0], recall


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def dataset(x, y):
    return SimpleDataset(x, y)


