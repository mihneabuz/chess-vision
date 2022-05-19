import torch
from torchvision import models, transforms
from torch.utils.data import random_split, DataLoader

from load_data import load_data
from utils import get_device, train_loop, validation_metrics, summary, dataset

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def tensor_transform(image):
    return torch.from_numpy(image.transpose(2, 0, 1))

def jit_transform(x):
    return normalize(x / 255)

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
    model = models.efficientnet_b0(pretrained=pretrained)
    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=8))
    return model

def loss_func(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    return (grouped - real).pow(2).sum(2).sqrt().sum()

def train(limit=-1, lr=0.0001, epochs=1, batch_size=8):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    # from cv2 import cv2
    # import matplotlib.pyplot as plt
    # for image, corners in train_ds:
    #     im = image.numpy().transpose(1, 2, 0)

    #     cv2.circle(im, [int(corners[0][1] * 1280), int(corners[0][0] * 1280)], 20, (0, 0, 255), 8)
    #     cv2.circle(im, [int(corners[1][1] * 1280), int(corners[1][0] * 1280)], 20, (255, 0, 0), 8)
    #     cv2.circle(im, [int(corners[2][1] * 1280), int(corners[2][0] * 1280)], 20, (0, 255, 255), 8)
    #     cv2.circle(im, [int(corners[3][1] * 1280), int(corners[3][0] * 1280)], 20, (0, 255, 0), 8)

    #     plt.imshow(im)
    #     plt.show()
           
    model = create_model()
    # summary(model, (3, 1280, 1280))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_func

    losses = []
    for i in range(epochs):
        print(f'epoch {i + 1}')
        curr_losses = train_loop(model, train_dl, optimizer, criterion, transform=jit_transform)
        print(f'loss: {sum(curr_losses)}')
        losses += curr_losses

if __name__ == '__main__':
    train(limit=10, epochs=4)
