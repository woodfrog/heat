import numpy as np
import torch
from torch.utils.data import Dataset
import os
import skimage
from scipy.ndimage import gaussian_filter
import cv2
from utils.nn_utils import positional_encoding_2d
from torchvision import transforms
from PIL import Image
from datasets.data_util import RandomBlur
import itertools
from torch.utils.data.dataloader import default_collate

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs


class BuildingCornerDataset(Dataset):
    def __init__(self, data_path, det_path, phase='train', image_size=256, rand_aug=True, d_pe=128, training_split=None,
                 inference=False):
        super(BuildingCornerDataset, self).__init__()
        self.data_path = data_path
        self.det_path = det_path
        self.phase = phase
        self.d_pe = d_pe
        self.rand_aug = rand_aug
        self.image_size = image_size
        self.inference = inference

        blur_transform = RandomBlur()
        self.train_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.3),
            blur_transform])

        if phase == 'train':
            datalistfile = os.path.join(data_path, 'train_list.txt')
            self.training = True
        else:
            datalistfile = os.path.join(data_path, 'valid_list.txt')
            self.training = False
        with open(datalistfile, 'r') as f:
            self._data_names = f.readlines()
        num_examples = len(self._data_names)
        if phase == 'train':
            if training_split == 'first':
                self._data_names = self._data_names[:num_examples // 2]
            elif training_split == 'second':
                self._data_names = self._data_names[num_examples // 2:]
        else:
            if phase == 'valid':
                self._data_names = self._data_names[:50]
            elif phase == 'test':
                self._data_names = self._data_names[50:]
            else:
                raise ValueError('Invalid phase {}'.format(phase))

    def __len__(self):
        return len(self._data_names)

    def __getitem__(self, idx):
        data_name = self._data_names[idx][:-1]
        annot_path = os.path.join(self.data_path, 'annot', data_name + '.npy')
        annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()
        det_path = os.path.join(self.det_path, data_name + '.npy')
        det_corners = np.array(np.load(det_path, allow_pickle=True))  # [N, 2]
        det_corners = det_corners[:, ::-1]  # turn into x,y format

        img_path = os.path.join(self.data_path, 'rgb', data_name + '.jpg')
        rgb = cv2.imread(img_path)

        if self.image_size != 256:
            rgb, annot, det_corners = self.resize_data(rgb, annot, det_corners)

        if self.rand_aug:
            image, annot, corner_mapping, det_corners = self.random_aug_annot(rgb, annot, det_corners=det_corners)
        else:
            image = rgb
        rec_mat = None

        corners = np.array(list(annot.keys()))[:, [1, 0]]

        if not self.inference and len(corners) > 100:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        if self.training:
            # Add some randomness for g.t. corners
            corners += np.random.normal(0, 0, size=corners.shape)
            pil_img = Image.fromarray(image)
            image = self.train_transform(pil_img)
            image = np.array(image)
        image = skimage.img_as_float(image)

        # sort by the second value and then the first value, here the corners are in the format of (y, x)
        sort_idx = np.lexsort(corners.T)
        corners = corners[sort_idx]

        corner_list = []
        for corner_i in range(corners.shape[0]):
            corner_list.append((corners[corner_i][1], corners[corner_i][0]))  # to (x, y) format

        raw_data = {
            'name': data_name,
            'corners': corner_list,
            'annot': annot,
            'image': image,
            'rec_mat': rec_mat,
            'annot_path': annot_path,
            'det_path': det_path,
            'img_path': img_path,
        }

        return self.process_data(raw_data)

    def process_data(self, data):
        img = data['image']
        corners = data['corners']
        annot = data['annot']
        rec_mat = data['rec_mat']

        # pre-process the image to use ImageNet-pretrained backbones
        img = img.transpose((2, 0, 1))
        raw_img = img.copy()
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)

        corners = np.array(corners)

        # corner labels
        pixel_labels, gauss_labels = self.get_corner_labels(corners)

        return {
            'pixel_labels': pixel_labels,
            'gauss_labels': gauss_labels,
            "annot": annot,
            "name": data['name'],
            'img': img,
            'raw_img': raw_img,
            'rec_mat': rec_mat,
            'annot_path': data['annot_path'],
            'det_path': data['det_path'],
            'img_path': data['img_path'],
        }

    def get_corner_labels(self, corners):
        labels = np.zeros((self.image_size, self.image_size))
        corners = corners.round()
        xint, yint = corners[:, 0].astype(np.int), corners[:, 1].astype(np.int)
        labels[yint, xint] = 1

        gauss_labels = gaussian_filter(labels, sigma=2)
        gauss_labels = gauss_labels / gauss_labels.max()
        return labels, gauss_labels

    def resize_data(self, image, annot, det_corners):
        new_image = cv2.resize(image, (self.image_size, self.image_size))
        new_annot = {}
        r = self.image_size / 256
        for c, connections in annot.items():
            new_c = tuple(np.array(c) * r)
            new_connections = [other_c * r for other_c in connections]
            new_annot[new_c] = new_connections
        new_dets = det_corners * r
        return new_image, new_annot, new_dets

    def random_aug_annot(self, img, annot, det_corners=None):
        # do random flipping
        img, annot, det_corners = self.random_flip(img, annot, det_corners)

        # prepare random augmentation parameters (only do random rotation for now)
        theta = np.random.randint(0, 360) / 360 * np.pi * 2
        r = self.image_size / 256
        origin = [127 * r, 127 * r]
        p1_new = [127 * r + 100 * np.sin(theta) * r, 127 * r - 100 * np.cos(theta) * r]
        p2_new = [127 * r + 100 * np.cos(theta) * r, 127 * r + 100 * np.sin(theta) * r]
        p1_old = [127 * r, 127 * r - 100 * r]  # y_axis
        p2_old = [127 * r + 100 * r, 127 * r]  # x_axis
        pts1 = np.array([origin, p1_old, p2_old]).astype(np.float32)
        pts2 = np.array([origin, p1_new, p2_new]).astype(np.float32)
        M_rot = cv2.getAffineTransform(pts1, pts2)

        # Combine annotation corners and detection corners
        all_corners = list(annot.keys())
        if det_corners is not None:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        all_corners_ = np.array(all_corners)

        # Do the per-corner transform
        # Done in a big matrix transformation to save processing time.
        corner_mapping = dict()
        ones = np.ones([all_corners_.shape[0], 1])
        all_corners_ = np.concatenate([all_corners_, ones], axis=-1)
        aug_corners = np.matmul(M_rot, all_corners_.T).T

        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = aug_corners[idx]

        # If the transformed geometry goes beyond image boundary, we simply re-do the augmentation
        new_corners = np.array(list(corner_mapping.values()))
        if new_corners.min() <= 0 or new_corners.max() >= (self.image_size - 1):
            # return self.random_aug_annot(img, annot, det_corners)
            return img, annot, None, det_corners

        # build the new annot dict
        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = list()
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        # Also transform the image correspondingly
        rows, cols, ch = img.shape
        new_img = cv2.warpAffine(img, M_rot, (cols, rows), borderValue=(255, 255, 255))

        y_start = (new_img.shape[0] - self.image_size) // 2
        x_start = (new_img.shape[1] - self.image_size) // 2
        aug_img = new_img[y_start:y_start + self.image_size, x_start:x_start + self.image_size, :]

        if det_corners is None:
            return aug_img, aug_annot, corner_mapping, None
        else:
            aug_det_corners = list()
            for corner in det_corners:
                new_corner = corner_mapping[tuple(corner)]
                aug_det_corners.append(new_corner)
            aug_det_corners = np.array(aug_det_corners)
            return aug_img, aug_annot, corner_mapping, aug_det_corners

    def random_flip(self, img, annot, det_corners):
        height, width, _ = img.shape
        rand_int = np.random.randint(0, 4)
        if rand_int == 0:
            return img, annot, det_corners

        all_corners = list(annot.keys())
        if det_corners is not None:
            for i in range(det_corners.shape[0]):
                all_corners.append(tuple(det_corners[i]))
        new_corners = np.array(all_corners)

        if rand_int == 1:
            img = img[:, ::-1, :]
            new_corners[:, 0] = width - new_corners[:, 0]
        elif rand_int == 2:
            img = img[::-1, :, :]
            new_corners[:, 1] = height - new_corners[:, 1]
        else:
            img = img[::-1, ::-1, :]
            new_corners[:, 0] = width - new_corners[:, 0]
            new_corners[:, 1] = height - new_corners[:, 1]

        new_corners = np.clip(new_corners, 0, self.image_size - 1)  # clip into [0, 255]
        corner_mapping = dict()
        for idx, corner in enumerate(all_corners):
            corner_mapping[corner] = new_corners[idx]

        aug_annot = dict()
        for corner, connections in annot.items():
            new_corner = corner_mapping[corner]
            tuple_new_corner = tuple(new_corner)
            aug_annot[tuple_new_corner] = list()
            for to_corner in connections:
                aug_annot[tuple_new_corner].append(corner_mapping[tuple(to_corner)])

        if det_corners is not None:
            aug_det_corners = list()
            for corner in det_corners:
                new_corner = corner_mapping[tuple(corner)]
                aug_det_corners.append(new_corner)
            det_corners = np.array(aug_det_corners)

        return img, aug_annot, det_corners


def collate_fn_corner(data):
    batched_data = {}
    for field in data[0].keys():
        if field in ['annot', 'rec_mat']:
            batch_values = [item[field] for item in data]
        else:
            batch_values = default_collate([d[field] for d in data])
        if field in ['pixel_features', 'pixel_labels', 'gauss_labels']:
            batch_values = batch_values.float()
        batched_data[field] = batch_values

    return batched_data


def get_pixel_features(image_size, d_pe=128):
    all_pe = positional_encoding_2d(d_pe, image_size, image_size)
    pixels_x = np.arange(0, image_size)
    pixels_y = np.arange(0, image_size)

    xv, yv = np.meshgrid(pixels_x, pixels_y)
    all_pixels = list()
    for i in range(xv.shape[0]):
        pixs = np.stack([xv[i], yv[i]], axis=-1)
        all_pixels.append(pixs)
    pixels = np.stack(all_pixels, axis=0)

    pixel_features = all_pe[:, pixels[:, :, 1], pixels[:, :, 0]]
    pixel_features = pixel_features.permute(1, 2, 0)
    return pixels, pixel_features


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DATAPATH = './data/cities_dataset'
    DET_PATH = './data/det_final'
    train_dataset = BuildingCornerDataset(DATAPATH, DET_PATH, phase='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0,
                                  collate_fn=collate_fn_corner)
    for i, item in enumerate(train_dataloader):
        import pdb;

        pdb.set_trace()
        print(item)
