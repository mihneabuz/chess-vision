import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from collections import Counter
from random import randint

from utils.load_data import load_data
from utils.process import crop_board, crop_pieces
from utils.utils import classes_dict, image_from_bytes, bytes_as_file, deserialize_array, num_classes, get_device, serialize_values, train_loop, validation_metrics, summary, dataset
import service as service

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
augments = transforms.Compose([
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomHorizontalFlip(0.5)
])

def tensor_transform(image):
    return torch.from_numpy(image.transpose(2, 0, 1))

def jit_transform(x):
    return normalize(x / 255)

def augment(x):
    return augments(jit_transform(x))

def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def load_board_images(max=-1):
    for image, annotations in load_data(max=max):
        yield image, annotations['corners']

def load_piece_images(max=-1):
    for image, annotations in load_data(max=max):
        board_image = crop_board(image, annotations['corners'])
        pieces, labels = crop_pieces(board_image, annotations['config'])
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
        drop_chance = avg_count * 150 // counts['empty']
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

    return random_split(dataset(pieces_images, pieces_classes), [train_size, valid_size])

def create_model(pretrained=True):
    if pretrained:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    else:
        model = models.efficientnet_b2()
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=num_classes))
    return model

def load_model():
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load('./piece_classification_weights'))
    return model

def inference():
    model = load_model()
    model.eval()
    return lambda img: model(jit_transform(tensor_transform(img))[None, :, :, :]).detach().numpy()

def train(epochs, lr=0.0001, batch_size=64, limit=-1, load_dict=False):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    if load_dict:
        model = load_model()
    else:
        model = create_model()

    model.to(device)
    summary(model, (3, 128, 128))

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
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=augment)
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

    torch.save(model.state_dict(), './piece_classification_weights');

class Service(service.Service):
    def __init__(self):
        self.name = 'piece_classification'
        self.model = create_model(pretrained=False);

    def load_model(self, data):
        self.model.load_state_dict(torch.load(bytes_as_file(data), map_location=torch.device('cpu')))

    def _transform_in(self, input):
        image = image_from_bytes(input[0])
        corners = deserialize_array(input[1])
        pieces, _ = crop_pieces(crop_board(image, np.array_split(corners, 4)))
        return torch.stack([jit_transform(tensor_transform(piece)) for piece in pieces])

    def _transform_out(self, result):
        preds = result.detach().numpy().astype(np.uint8)
        return serialize_values(preds)

    def _process_batch(self, data):
        return torch.argmax(self.model(torch.concat(data)), 1).split(64)

if __name__ == "__main__":
    train(6, batch_size=32, limit=-1, load_dict=False)
