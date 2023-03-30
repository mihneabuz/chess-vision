import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import random_split, DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.load_data import load_data
from utils.utils import get_device, train_loop, dataset, serialize_array, bytes_as_file, image_from_bytes, summary
import service as service

size = 224

def tensor_transform(image):
    resized = cv2.resize(image, (size, size))
    return torch.from_numpy(resized.transpose(2, 0, 1))

def jit_transform(x):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    return normalize(x / 255)

def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def load_datasets(limit=-1):
    images = []
    corners = []
    for image, annotations in load_data(max=limit):
        images.append(tensor_transform(image))
        corners.append(torch.tensor(annotations['corners']))

    print(f'loaded {len(images)} images')
    print(f'image size: {images[0].shape}')

    size = len(images)
    train_size = int(size * 0.8)
    valid_size = size - train_size

    return random_split(dataset(images, corners), [train_size, valid_size])

def create_model():
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=8))
    return model

def load_model():
    model = create_model()
    model.load_state_dict(torch.load('./board_detection_weights'))
    return model

def old_loss_func(predicted, real):
    grouped1 = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    grouped2 = torch.stack((predicted[:, 6:8], predicted[:, 4:6], predicted[:, 2:4], predicted[:, 0:2]), 1)

    dists1 = (grouped1 - real).pow(2).sum(2).sqrt().sum(1)
    dists2 = (grouped2 - real).pow(2).sum(2).sqrt().sum(1)

    return torch.min(dists1, dists2).sum()

def loss_func_sum(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    return (grouped - real).pow(2).sum(2).sqrt().sum(1).sum()

def loss_func_max(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    maxss, _ = torch.max((grouped - real).pow(2).sum(2).sqrt(), dim=1)
    return maxss.sum()

def loss_func_sum_diag(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 6:8]), 1)
    real = torch.stack((real[:, 0], real[:, 3]), 1)
    return (grouped - real).pow(2).sum(2).sqrt().sum(1).sum()

def train(epochs, lr=0.0001, batch_size=4, limit=-1, load_dict=False):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2)

    i = 0
    for image, corners in train_ds:
        im = image.numpy().transpose(1, 2, 0)

        cv2.circle(im, (int(corners[0][1] * size), int(corners[0][0] * size)), 2, (255, 0, 0), 2)
        cv2.circle(im, (int(corners[1][1] * size), int(corners[1][0] * size)), 2, (0, 255, 0), 2)
        cv2.circle(im, (int(corners[2][1] * size), int(corners[2][0] * size)), 2, (0, 0, 255), 2)
        cv2.circle(im, (int(corners[3][1] * size), int(corners[3][0] * size)), 2, (0, 255, 255), 2)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 3):
            break

    if load_dict:
        model = load_model()
    else:
        model = create_model()

    model.to(device)
    summary(model, (3, size, size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = loss_func_sum

    if epochs > 10:
        set_grad(model.features, False)
        for i in range(5):
            curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
            print(f'loss: {sum(curr_losses)}')
        set_grad(model.features, True)

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

    plt.plot(losses)
    plt.show()

    model.eval()
    valid_losses = []
    for images, labels in tqdm(valid_dl, desc='validation'):
        preds = model(jit_transform(images.to(device)))
        valid_losses.append(criterion(preds, labels.to(device)).item())
    print(f'validation loss: {sum(valid_losses)}')

    i = 0
    for image, corners in valid_ds:
        preds = model(jit_transform(image)[None, :, :, :].to(device))
        im = image.numpy().transpose(1, 2, 0)

        cv2.circle(im, [int(corners[0][1] * size), int(corners[0][0] * size)], 2, (255, 0, 0), 2)
        cv2.circle(im, [int(corners[1][1] * size), int(corners[1][0] * size)], 2, (0, 255, 0), 2)
        cv2.circle(im, [int(corners[2][1] * size), int(corners[2][0] * size)], 2, (0, 0, 255), 2)
        cv2.circle(im, [int(corners[3][1] * size), int(corners[3][0] * size)], 2, (0, 255, 255), 2)

        cv2.circle(im, [int(preds[0][1] * size), int(preds[0][0] * size)], 6, (255, 0, 0), 2)
        cv2.circle(im, [int(preds[0][3] * size), int(preds[0][2] * size)], 6, (0, 255, 0), 2)
        cv2.circle(im, [int(preds[0][5] * size), int(preds[0][4] * size)], 6, (0, 0, 255), 2)
        cv2.circle(im, [int(preds[0][7] * size), int(preds[0][6] * size)], 6, (0, 255, 255), 2)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 20):
            break

    torch.save(model.state_dict(), './board_detection_weights');

class Service(service.Service):
    def __init__(self):
        self.name = 'board_detection'
        self.model = create_model();

    def load_model(self, data):
        self.model.load_state_dict(torch.load(bytes_as_file(data), map_location=torch.device('cpu')))

    def _transform_in(self, input):
        image = image_from_bytes(input[0])
        return jit_transform(tensor_transform(image))

    def _transform_out(self, result):
        return serialize_array(result.astype(np.float32))

    def _process_batch(self, data):
        return self.model(torch.stack(data)).detach().numpy()

if __name__ == '__main__':
    train(50, lr=0.00001, batch_size=10, load_dict=False, limit=-1)
