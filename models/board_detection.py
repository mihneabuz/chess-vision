import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import models
from torch.utils.data import DataLoader
import cv2
import random

import service as service
from utils.load_data import load_data
from utils.corners import sort_corners
from utils.utils import get_device, train_loop, dataset, serialize_array, bytes_as_file, image_from_bytes, summary

size = 200


def augment(image, corners):
    if random.random() > 0.4:
        image = TF.adjust_hue(image, random.randint(-1, 1) / 10)
        image = TF.adjust_contrast(image, 1 + random.randint(-3, 3) / 10)

    if random.random() > 1.6:
        amount = random.randint(4, 40)
        cropped = TF.center_crop(image, (size - amount, size - amount))
        image = TF.resize(cropped, (size, size), antialias=None)

        percent = size / (size - amount) - 1
        corners[corners >= 0.5] += (corners[corners >= 0.5] - 0.5) * percent
        corners[corners < 0.5] -= (0.5 - corners[corners < 0.5]) * percent

    if random.random() > 0.9:
        image = TF.gaussian_blur(image, 3)

    return image, corners


def tensor_transform(image):
    resized = cv2.resize(image, (size, size))
    return torch.from_numpy(resized.transpose(2, 0, 1))


def jit_transform(x):
    return TF.normalize(x / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def load_datasets(limit=-1):
    images = []
    corners = []
    for image, annotations in load_data(max=limit):
        images.append(tensor_transform(image))
        corns = sort_corners(list(map(lambda x: [x[0], 1 - x[1]], annotations['corners'])))
        corners.append(torch.tensor(corns))

    print(f'loaded {len(images)} images')
    print(f'image size: {images[0].shape}')

    count = len(images)
    train_size = int(count * 0.8)
    valid_size = count - train_size

    from sklearn.model_selection import train_test_split
    train_im, test_im, train_mk, test_mk = train_test_split(images, corners, test_size=valid_size)
    train = dataset(train_im, train_mk, augment=augment)
    test = dataset(test_im, test_mk)

    return train, test


def create_model(load_dict=False, pretrained=False):
    if pretrained:
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_v2_m()
        if load_dict:
            model.load_state_dict(torch.load('./board_detection_weights'))

    last_layer_size = model.classifier[-1].__getattribute__('out_features')
    model.classifier.append(torch.nn.Linear(in_features=last_layer_size, out_features=8))
    return model


def loss_func_sum(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    return (grouped - real).pow(2).sum(2).sqrt().sum(1).sum()


def loss_func_max(predicted, real):
    grouped = torch.stack((predicted[:, 0:2], predicted[:, 2:4], predicted[:, 4:6], predicted[:, 6:8]), 1)
    maxss, _ = torch.max((grouped - real).pow(2).sum(2).sqrt(), dim=1)
    return maxss.sum()


def train(epochs, lr=0.0001, batch_size=4, limit=-1, load_dict=False):
    import matplotlib.pyplot as plt

    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2)

    i = 0
    for image, corners in train_ds:
        im = image.numpy().transpose(1, 2, 0).copy()
        c = (corners.numpy() * size).astype(np.int32)

        cv2.circle(im, (c[0][0], c[0][1]), 2, (255, 0, 0), 2)
        cv2.circle(im, (c[1][0], c[1][1]), 2, (0, 255, 0), 2)
        cv2.circle(im, (c[2][0], c[2][1]), 2, (0, 0, 255), 2)
        cv2.circle(im, (c[3][0], c[3][1]), 2, (0, 255, 255), 2)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 5):
            break

    model = create_model(load_dict=load_dict, pretrained=not load_dict)
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
    for images, labels in valid_dl:
        preds = model(jit_transform(images.to(device)))
        valid_losses.append(criterion(preds, labels.to(device)).item())
    print(f'validation loss: {sum(valid_losses)}')

    i = 0
    for image, corners in valid_ds:
        preds = model(jit_transform(image)[None, :, :, :].to(device))
        im = image.numpy().transpose(1, 2, 0)

        c = (corners.numpy() * size).astype(np.int32)
        cv2.circle(im, (c[0][0], c[0][1]), 2, (255, 0, 0), 2)
        cv2.circle(im, (c[1][0], c[1][1]), 2, (0, 255, 0), 2)
        cv2.circle(im, (c[2][0], c[2][1]), 2, (0, 0, 255), 2)
        cv2.circle(im, (c[3][0], c[3][1]), 2, (0, 255, 255), 2)

        p = (preds.detach().cpu().numpy() * size).astype(np.int32)[0]
        cv2.circle(im, (p[0], p[1]), 6, (255, 0, 0), 2)
        cv2.circle(im, (p[2], p[3]), 6, (0, 255, 0), 2)
        cv2.circle(im, (p[4], p[5]), 6, (0, 0, 255), 2)
        cv2.circle(im, (p[6], p[7]), 6, (0, 255, 255), 2)

        plt.imshow(im)
        plt.show()

        i += 1
        if (i > 20):
            break

    if not load_dict:
        torch.save(model.state_dict(), './board_detection_weights')


class Service(service.Service):
    def __init__(self):
        self.name = 'board_detection'
        self.model = create_model()
        self.model.eval()

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
    train(30, lr=0.00003, batch_size=20, load_dict=False, limit=-1)
