import numpy as np
import cv2
from io import BytesIO
import torch
import torchsummary
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

classes_dict = {
    'empty':    0,
    'pawn_w':   1,
    'pawn_b':   2,
    'bishop_w': 3,
    'bishop_b': 4,
    'knight_w': 5,
    'knight_b': 6,
    'rook_w':   7,
    'rook_b':   8,
    'queen_w':  9,
    'queen_b': 10,
    'king_w':  11,
    'king_b':  12,
}

num_classes = len(classes_dict)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def summary(model, input):
    torchsummary.summary(model, input_size=input, device='cuda' if torch.cuda.is_available() else 'cpu')


def id1(x):
    return x


def id2(x, y):
    return x, y


def train_loop(model, dataloader, optimizer, criterion, transform=id1):
    device = get_device()
    model.train(True)

    losses = []
    for images, labels in tqdm(dataloader, desc='batches'):
        images = transform(images.to(device))
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels)

        losses.append(loss.item() / len(images))

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
    def __init__(self, x, y, augment=id2):
        self.data = list(zip(x, y))
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.augment(*self.data[idx])


def dataset(x, y, augment=id2):
    return SimpleDataset(x, y, augment=augment)


def bytes_as_file(bytes: bytes) -> BytesIO:
    memfile = BytesIO()
    memfile.write(bytes)
    memfile.seek(0)
    return memfile


def image_from_bytes(bytes: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(bytes, np.uint8), cv2.IMREAD_COLOR)


def serialize_array(ndarray: np.ndarray) -> bytes:
    memfile = BytesIO()
    np.save(memfile, ndarray)
    return memfile.getvalue()


def deserialize_array(bytes: bytes) -> np.ndarray:
    memfile = bytes_as_file(bytes)
    return np.load(memfile, allow_pickle=True)


def serialize_values(values) -> bytes:
    memfile = BytesIO()
    for value in values:
        memfile.write(value)
    return memfile.getvalue()


def deserialize_values(bytes: bytes, count: int, dtype):
    memfile = bytes_as_file(bytes)
    values = [int.from_bytes(memfile.read(np.dtype(dtype).itemsize), "big") for _ in range(count)]
    return np.array(values, dtype=dtype)
