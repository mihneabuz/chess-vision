import numpy as np
import torch
from torch import nn
from torchvision import models
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from random import randint

from load_data import load_data, get_corners, get_pieces
from crop_board import batch_crop_board, batch_crop_pieces

GENERATOR = torch.Generator().manual_seed(42)

class PieceDataset(Dataset):
    def __init__(self, images, labels):
        device = get_device()
        self.images = [torch.tensor(image.transpose(2, 0, 1)).to(device) / 255 for image in images]
        self.labels = [torch.tensor(label).to(device) for label in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

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

def load_datasets(limit=-1, balance=True):
    images, labels = load_data(max=limit, gray=False)
    corners = get_corners(labels)
    pieces = get_pieces(labels)

    board_images = batch_crop_board(images, corners)
    pieces_images, pieces_labels = batch_crop_pieces(board_images, pieces)

    image_size = pieces_images[0].shape
    print(f'image size: {image_size}')

    counts = Counter(pieces_labels)
    print(f'label distribution: {", ".join([x[0] + ": " + str(x[1]) for x in counts.items()])}')
    plt.bar(counts.keys(), counts.values())
    plt.show()

    if balance:
        avg_count = sum([counts[label] for label in classes_dict if label != 'empty']) // 6
        drop_chance = avg_count * 100 // counts['empty']
        dropped = 0

        balanced_pieces_images = []
        balanced_pieces_labels = []

        for image, label in zip(pieces_images, pieces_labels):
            if label == 'empty':
                if randint(0, 100) < drop_chance:
                    balanced_pieces_images.append(image)
                    balanced_pieces_labels.append(label)
                else:
                    dropped += 1
            else:
                balanced_pieces_images.append(image)
                balanced_pieces_labels.append(label)
        print(f'dropped {dropped} empty images')

        counts = Counter(balanced_pieces_labels)
        print(f'balanced label distribution: {", ".join([x[0] + ": " + str(x[1]) for x in counts.items()])}')
        plt.bar(counts.keys(), counts.values())
        plt.show()

        pieces_images = balanced_pieces_images
        pieces_labels = balanced_pieces_labels

    pieces_classes = [classes_dict[label] for label in pieces_labels]
    data = PieceDataset(pieces_images, pieces_classes) 

    size = len(data)
    train_size = int(size * 0.7)
    valid_size = size - train_size

    train_ds, valid_ds = random_split(data, [train_size, valid_size], generator=GENERATOR)
    return train_ds, valid_ds

def train_loop(model, dataloader, optimizer, criterion):
    model.train()

    losses = []
    for data in tqdm(dataloader, desc='batches'):
        images, labels = data

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return losses

def validation_metrics(model, dataloader):
    preds = np.array([])
    real = np.array([])

    for data in tqdm(dataloader, desc='accuracy metrics'):
        images, labels = data

        preds = np.concatenate((preds, torch.argmax(model(images), dim=1).numpy()))
        real = np.concatenate((real, labels.numpy()))

    accuracy = accuracy_score(real, preds),
    f1 = f1_score(real, preds, labels=range(num_classes), average='macro'),

    precision = precision_score(real, preds, labels=range(num_classes), average=None),
    recall = recall_score(real, preds, labels=range(num_classes), average=None)

    return accuracy[0], f1[0], precision[0], recall
           

def train():
    train_ds, valid_ds = load_datasets(10)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2)

    model = models.efficientnet_b2(pretrained=True)
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(nn.Linear(in_features=last_layer_size, out_features=num_classes))
    summary(model, input_size=(3, 128, 128))

    device = get_device()
    model.to(device)

    epochs = 2
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

    plt.plot(losses)
    plt.show()

    accuracy, f1, precision, recall = validation_metrics(model, valid_dl)
    print(f'accuracy: {accuracy}\nf1 score: {f1}')

    print('precision / recall')
    for i, label in enumerate(classes_dict.keys()):
        print(f'{label:10} {precision[i]:.2f} {recall[i]:.2f}')

if __name__ == "__main__":
    train()
