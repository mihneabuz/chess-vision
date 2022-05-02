import torch
from torchvision import models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
from random import randint

from load_data import load_piece_images
from utils import classes_dict, num_classes, get_device, train_loop, validation_metrics, summary, dataset

GEN = torch.Generator()

def tensor_transform(image):
    return torch.from_numpy(image.transpose(2, 0, 1))

def jit_transform(x):
    return x / 255

def load_datasets(limit=-1, balance=True):
    pieces_images = []
    pieces_labels = []
    for image, label in load_piece_images(max=limit):
        pieces_images.append(tensor_transform(image))
        pieces_labels.append(label)

    image_size = pieces_images[0].shape
    print(f'image size: {image_size}')

    counts = Counter(pieces_labels)
    print(f'label distribution: {", ".join([x[0] + ": " + str(x[1]) for x in counts.items()])}')
    plt.bar(counts.keys(), counts.values())
    plt.show()

    if balance:
        avg_count = sum([counts[label] for label in classes_dict if label != 'empty']) // 10
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

    size = len(pieces_images)
    train_size = int(size * 0.7)
    valid_size = size - train_size

    return random_split(dataset(pieces_images, pieces_classes), [train_size, valid_size], generator=GEN)

def train(epochs, batch_size=64, limit=-1):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    model = models.efficientnet_b2(pretrained=True)
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=num_classes))
    model.to(device)
    summary(model, (3, 128, 128))

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

    plt.plot(losses)
    plt.show()

    metrics = validation_metrics(model, valid_dl, transform=jit_transform, results=lambda preds: torch.argmax(preds, dim=1))
    accuracy, f1, precision, recall = metrics
    print(f'accuracy: {accuracy:.4f}\nf1 score: {f1:.4f}')

    print('precision / recall')
    for i, label in enumerate(classes_dict.keys()):
        print(f'{label:10} {precision[i]:.2f} {recall[i]:.2f}')

if __name__ == "__main__":
    train(2, batch_size=32, limit=10)
