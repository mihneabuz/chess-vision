import numpy as np
import torch
import service as service
from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import random

from utils.load_data import load_data
from utils.process import crop_board, crop_pieces
from utils.utils import classes_dict, image_from_bytes, bytes_as_file, deserialize_array,\
    num_classes, get_device, serialize_values, train_loop, validation_metrics, summary, dataset

size = 144
growthFactor = 0.08


def augment(image, label):
    size = image.shape[1]

    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)

    if random.random() > 0.4:
        image = TF.adjust_hue(image, random.randint(-1, 1) / 10)
        image = TF.adjust_contrast(image, 1 + random.randint(-3, 3) / 10)

    if random.random() > 0.8:
        amount = random.randint(4, 24)
        cropped = TF.center_crop(image, (size - amount, size - amount))
        image = TF.resize(cropped, (size, size), antialias=None)

    if random.random() > 0.9:
        image = TF.gaussian_blur(image, 3)

    if random.random() > 0.5:
        image = TF.hflip(image)

    if random.random() > 0.5:
        image = TF.vflip(image)

    return image, label


def tensor_transform(image):
    return torch.from_numpy(image.transpose(2, 0, 1))


def jit_transform(x):
    return TF.normalize(x / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def create_model(load_dict=False, pretrained=False):
    if pretrained:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    else:
        model = models.efficientnet_b2()
        if load_dict:
            model.load_state_dict(torch.load('./piece_classification_weights'))

    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=num_classes))
    return model


def results(preds):
    return torch.argmax(preds, dim=1)


def load_board_images(max=-1):
    for image, annotations in load_data(max=max):
        yield image, annotations['corners']


def load_piece_images(max=-1):
    for image, annotations in load_data(max=max):
        board_image = crop_board(image, annotations['corners'], growthFactor=growthFactor, imageSize=size)
        pieces, labels = crop_pieces(board_image, annotations['config'], growthFactor=growthFactor, imageSize=size)
        yield from zip(pieces, labels)


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
        drop_chance = avg_count * 300 // counts['empty']
        dropped = 0

        balanced_pieces_images = []
        balanced_pieces_labels = []

        for image, label in zip(pieces_images, pieces_labels):
            if label == 'empty':
                if random.randint(0, 100) < drop_chance:
                    balanced_pieces_images.append(image)
                    balanced_pieces_labels.append(label)
                else:
                    dropped += 1
            else:
                balanced_pieces_images.append(image)
                balanced_pieces_labels.append(label)
        print(f'dropped {dropped} empty images')

        counts = Counter(balanced_pieces_labels)
        print(
            f'balanced label distribution: {", ".join([x[0] + ": " + str(x[1]) for x in counts.items()])}')
        plt.bar(counts.keys(), counts.values())
        plt.show()

        pieces_images = balanced_pieces_images
        pieces_labels = balanced_pieces_labels

    pieces_classes = [classes_dict[label] for label in pieces_labels]

    size = len(pieces_images)
    train_size = int(size * 0.8)
    valid_size = size - train_size

    train_im, test_im, train_mk, test_mk = train_test_split(pieces_images, pieces_classes, test_size=valid_size)
    train = dataset(train_im, train_mk, augment=augment)
    test = dataset(test_im, test_mk)

    return train, test


def train(epochs, lr=0.0001, batch_size=64, limit=-1, load_dict=False):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    for i, (image, label) in enumerate(train_ds):
        plt.imshow(image.numpy().transpose(1, 2, 0))
        plt.show()

        if (i > 3):
            break

    model = create_model(load_dict=load_dict, pretrained=not load_dict)
    model.to(device)
    summary(model, (3, size, size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print('training final layer')
    set_grad(model.features, False)
    train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
    train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
    set_grad(model.features, True)

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

    plt.plot(losses)
    plt.show()

    metrics = validation_metrics(model, valid_dl, transform=jit_transform, results=results)
    accuracy, f1, precision, recall = metrics
    print(f'accuracy: {accuracy:.4f}\nf1 score: {f1:.4f}')

    print('precision / recall')
    for i, label in enumerate(classes_dict.keys()):
        print(f'{label:10} {precision[i]:.2f} {recall[i]:.2f}')

    torch.save(model.state_dict(), './piece_classification_weights')


class Service(service.Service):
    def __init__(self):
        self.name = 'piece_classification'
        self.model = create_model(pretrained=False)

    def load_model(self, data):
        self.model.load_state_dict(torch.load(
            bytes_as_file(data), map_location=torch.device('cpu')))

    def _transform_in(self, input):
        image = image_from_bytes(input[0])
        corners = np.array_split(deserialize_array(input[1]), 4)

        try:
            board = crop_board(image, corners, flag=True, growthFactor=growthFactor, imageSize=size)
            pieces, _ = crop_pieces(board, pieces=None, growthFactor=growthFactor, imageSize=size)
            return torch.stack([jit_transform(tensor_transform(piece)) for piece in pieces])
        except Exception as e:
            print('bad corners', e, corners)
            return None

    def _transform_out(self, result):
        preds = result.detach().numpy().astype(np.uint8)
        return serialize_values(preds)

    def _process_batch(self, data):
        return torch.argmax(self.model(torch.concat(data)), 1).split(64)


if __name__ == "__main__":
    train(16, batch_size=32, limit=-1, load_dict=False)
