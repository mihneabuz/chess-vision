import torch
from torch import nn
from torchvision import models
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from load_data import load_data, get_corners, get_pieces
from crop_board import batch_crop_board, batch_crop_pieces

GENERATOR = torch.Generator().manual_seed(42)

class PieceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = [torch.tensor(image.transpose(2, 0, 1)) / 255 for image in images]
        self.labels = labels

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

def load_datasets(limit=-1):
    images, labels = load_data(max=limit, gray=False)
    corners = get_corners(labels)
    pieces = get_pieces(labels)

    board_images = batch_crop_board(images, corners)
    pieces_images, pieces_labels = batch_crop_pieces(board_images, pieces)

    image_size = pieces_images[0].shape
    counts = Counter(pieces_labels)
    pieces_classes = [classes_dict[label] for label in pieces_labels]

    print(f'image size: {image_size}')
    print(f'label distribution: {", ".join([x[0] + ": " + str(x[1]) for x in counts.items()])}')

    plt.bar(counts.keys(), counts.values())
    plt.show()

    # TODO: fix label distribution -> remove some empty cell pictures

    data = PieceDataset(pieces_images, pieces_classes) 
    size = len(data)
    train_size = int(size * 0.7)
    valid_size = size - train_size

    train_ds, valid_ds = random_split(data, [train_size, valid_size], generator=GENERATOR)
    return train_ds, valid_ds

def train_loop(model, dataloader, optimizer, criterion):
    model.train()

    total_loss = torch.tensor(0.)
    for data in tqdm(dataloader, desc='batches'):
        images, labels = data

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss

        loss.backward()
        optimizer.step()

    return total_loss.item()

def train():
    train_ds, valid_ds = load_datasets()
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    model = models.efficientnet_b2(pretrained=True)
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(nn.Linear(in_features=last_layer_size, out_features=num_classes))
    summary(model, input_size=(3, 1280, 1280))

    epochs = 4
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(epochs):
        print(f'epoch {i + 1}')
        loss = train_loop(model, train_dl, optimizer, criterion)
        print(f'loss: {loss}')

if __name__ == "__main__":
    train()
