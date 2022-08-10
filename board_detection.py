import torch
from torchvision import models, transforms
from torch.utils.data import random_split, DataLoader
import cv2
import matplotlib.pyplot as plt

from load_data import load_data
from utils import get_device, train_loop, dataset

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def tensor_transform(image):
    resized = cv2.resize(image, (640, 640))
    return torch.from_numpy(resized.transpose(2, 0, 1))

def jit_transform(x):
    y = normalize(x / 255)
    return y.half()

def load_datasets(limit=-1):
    images = []
    corners = []
    for image, annotations in load_data(max=limit):
        images.append(tensor_transform(image))
        corners.append(torch.tensor(annotations['corners']))

    print(f'loaded {len(images)} images')
    print(f'image size: {images[0].shape}')

    size = len(images)
    train_size = int(size * 0.7)
    valid_size = size - train_size

    return random_split(dataset(images, corners), [train_size, valid_size])

def create_model(pretrained=True):
    if pretrained:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    else:
        model = models.efficientnet_b3()
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=8))
    return model.half()

def load_model():
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load('./detection_weights'))
    return model

def loss_func(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    return (grouped - real).pow(2).sum(2).sqrt().sum()

def train(epochs, lr=0.001, batch_size=4, limit=-1, load_dict=False):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    i = 0
    for image, corners in train_ds:
        im = image.numpy().transpose(1, 2, 0)

        cv2.circle(im, (int(corners[0][1] * 640), int(corners[0][0] * 640)), 10, (0, 0, 255), 6)
        cv2.circle(im, (int(corners[1][1] * 640), int(corners[1][0] * 640)), 10, (255, 0, 0), 6)
        cv2.circle(im, (int(corners[2][1] * 640), int(corners[2][0] * 640)), 10, (0, 255, 255), 6)
        cv2.circle(im, (int(corners[3][1] * 640), int(corners[3][0] * 640)), 10, (0, 255, 0), 6)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 5):
            break

    if load_dict:
        model = load_model()
    else:
        model = create_model(pretrained=True)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), eps=1e-05, lr=lr)
    criterion = loss_func

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

    plt.plot(losses)
    plt.show()

    i = 0
    model.eval()
    for image, corners in valid_ds:
        preds = model(jit_transform(image)[None, :, :, :].to(device))
        im = image.numpy().transpose(1, 2, 0)

        cv2.circle(im, [int(corners[0][1] * 640), int(corners[0][0] * 640)], 10, (0, 0, 255), 4)
        cv2.circle(im, [int(corners[1][1] * 640), int(corners[1][0] * 640)], 10, (255, 0, 0), 4)
        cv2.circle(im, [int(corners[2][1] * 640), int(corners[2][0] * 640)], 10, (0, 255, 255), 4)
        cv2.circle(im, [int(corners[3][1] * 640), int(corners[3][0] * 640)], 10, (0, 255, 0), 4)

        cv2.circle(im, [int(preds[0][1] * 640), int(preds[0][0] * 640)], 15, (0, 0, 155), 4)
        cv2.circle(im, [int(preds[0][3] * 640), int(preds[0][2] * 640)], 15, (155, 0, 0), 4)
        cv2.circle(im, [int(preds[0][5] * 640), int(preds[0][4] * 640)], 15, (0, 155, 155), 4)
        cv2.circle(im, [int(preds[0][7] * 640), int(preds[0][6] * 640)], 15, (0, 155, 0), 4)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 10):
            break

    torch.save(model.state_dict(), './detection_weights');

if __name__ == '__main__':
    train(limit=10, lr=0.0001, epochs=4, load_dict=False)
