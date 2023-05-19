import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import service as service
from utils.load_data import load_data
from utils.corners import find_corners, debug_corners
from utils.utils import get_device, train_loop, dataset, bytes_as_file, image_from_bytes, summary, serialize_array

import u2net.model as u2net

size = 240


def augment(image, mask):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
        mask = TF.rotate(torch.unsqueeze(mask, 0), angle)[0, :, :]

    if random.random() > 0.4:
        image = TF.adjust_hue(image, random.randint(-1, 1) / 10)
        image = TF.adjust_contrast(image, 1 + random.randint(-3, 3) / 10)

    if random.random() > 0.8:
        amount = random.randint(4, 24)

        cropped = TF.center_crop(image, (size - amount, size - amount))
        image = TF.resize(cropped, (size, size), antialias=None)

        cropped = TF.center_crop(mask[None, :, :], (size - amount, size - amount))
        mask = TF.resize(cropped, (size, size), antialias=None)[0, :, :]

    if random.random() > 0.9:
        image = TF.gaussian_blur(image, 3)

    if random.random() > 0.4:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    if random.random() > 0.2:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    return image, mask


def tensor_transform(image):
    resized = cv2.resize(image, (size, size))
    return torch.from_numpy(resized.transpose(2, 0, 1))


def jit_transform(x):
    return TF.normalize(x / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def create_model(load_dict=False):
    model = u2net.U2NET()
    if load_dict:
        model.load_state_dict(torch.load('./board_segmentation_weights'))
    return model


def loss_func(preds, real):
    (d1, d2, d3, d4, d5, d6, d7) = preds

    l1 = (d1[:, 0, :, :] - real).abs().sum()
    l2 = (d2[:, 0, :, :] - real).abs().sum()
    l3 = (d3[:, 0, :, :] - real).abs().sum()
    l4 = (d4[:, 0, :, :] - real).abs().sum()
    l5 = (d5[:, 0, :, :] - real).abs().sum()
    l6 = (d6[:, 0, :, :] - real).abs().sum()
    l7 = (d7[:, 0, :, :] - real).abs().sum()

    scale = size * size

    return (l1 + l2 + l3 + l4 + l5 + l6 + l7) / scale


def load_datasets(limit=-1):
    images = []
    masks = []
    for image, annotations in load_data(max=limit):
        images.append(tensor_transform(image))

        mask = np.zeros((size, size), dtype=np.float32)
        corners = list(map(lambda x: [x[0], 1 - x[1]], annotations['corners']))
        corners = [corners[0], corners[1], corners[3], corners[2]]
        cv2.fillPoly(
            mask, pts=[(np.array(corners) * size).astype(np.int32)], color=1)
        masks.append(torch.tensor(mask))

    print(f'loaded {len(images)} images')
    print(f'image size: {images[0].shape}')

    count = len(images)
    train_size = int(count * 0.9)
    valid_size = count - train_size

    train_im, test_im, train_mk, test_mk = train_test_split(images, masks, test_size=valid_size)
    train = dataset(train_im, train_mk, augment=augment)
    test = dataset(test_im, test_mk)

    return train, test


def train(epochs, lr=0.0001, batch_size=4, limit=-1, load_dict=False):
    device = get_device()

    train_ds, valid_ds = load_datasets(limit=limit)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=2)

    for i, (image, mask) in enumerate(train_ds):
        im = image.numpy().transpose(1, 2, 0)
        mk = mask.numpy()
        im[mk == 0] = (im[mk == 0].astype(np.float32) * 0.3).astype(np.uint8)

        plt.imshow(im)
        plt.show()

        if (i > 5):
            break

    model = create_model(load_dict=load_dict)
    model.to(device)
    summary(model, (3, size, size))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_func

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

    for i, (image, mask) in enumerate(valid_ds):
        preds = model(jit_transform(image)[None, :, :, :].to(device))
        im = image.numpy().transpose(1, 2, 0)
        mk = (preds[0][0, 0, :, :].cpu().detach().numpy() * 255).astype(np.uint8)

        res = debug_corners(im, mk)
        plt.imshow(res)
        plt.show()

        if (i > 15):
            break

    if not load_dict:
        torch.save(model.state_dict(), './board_segmentation_weights')


class Service(service.Service):
    def __init__(self, maskSize=100, quality=0.3):
        self.name = 'board_segmentation'
        self.model = create_model(load_dict=False)
        self.model.eval()
        self.threshold = 0.9
        self.quality = quality
        self.maskSize = maskSize

    def load_model(self, data):
        self.model.load_state_dict(torch.load(bytes_as_file(data), map_location=torch.device('cpu')))

    def _transform_in(self, input):
        image = image_from_bytes(input[0])
        return jit_transform(tensor_transform(image))

    def _transform_out(self, result):
        mask = (result[0, :, :] > self.threshold).astype(np.uint8) * 255
        corners = find_corners(mask, downscale=self.maskSize, quality=self.quality).flatten()
        return serialize_array(corners) if len(corners) else None

    def _process_batch(self, data):
        return self.model(torch.stack(data))[0].detach().numpy()


if __name__ == '__main__':
    train(12, lr=0.0003, batch_size=8, limit=-1, load_dict=False)
