import numpy as np
from datasets.corners import CornersDataset
import os
import skimage
import cv2
import itertools


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

all_combibations = dict()
for length in range(2, 351):
    ids = np.arange(length)
    combs = np.array(list(itertools.combinations(ids, 2)))
    all_combibations[length] = combs


class S3DFloorplanDataset(CornersDataset):
    def __init__(self, data_path, phase='train', image_size=256, rand_aug=True, inference=False):
        super(S3DFloorplanDataset, self).__init__(image_size, inference)
        self.data_path = data_path
        self.phase = phase
        self.rand_aug = rand_aug

        if phase == 'train':
            datalistfile = os.path.join(data_path, 'train_list.txt')
            self.training = True
        elif phase == 'valid':
            datalistfile = os.path.join(data_path, 'valid_list.txt')
            self.training = False
        else:
            datalistfile = os.path.join(data_path, 'test_list.txt')
            self.training = False
        with open(datalistfile, 'r') as f:
            self._data_names = f.readlines()

    def __len__(self):
        return len(self._data_names)

    def __getitem__(self, idx):
        data_name = self._data_names[idx][:-1]
        annot_path = os.path.join(self.data_path, 'annot', data_name + '.npy')
        annot = np.load(annot_path, allow_pickle=True, encoding='latin1').tolist()

        density_path = os.path.join(self.data_path, 'density', data_name + '.png')
        normal_path = os.path.join(self.data_path, 'normals', data_name + '.png')

        density = cv2.imread(density_path)
        normal = cv2.imread(normal_path)
        rgb = np.maximum(density, normal)

        if self.image_size != 256:
            rgb, annot, det_corners = self.resize_data(rgb, annot, None)

        if self.rand_aug:
            image, annot, _ = self.random_aug_annot(rgb, annot, det_corners=None)
        else:
            image = rgb
        rec_mat = None

        corners = np.array(list(annot.keys()))[:, [1, 0]]

        if not self.inference and len(corners) > 150:
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)

        if self.training:
            # Add some randomness for g.t. corners
            corners += np.random.normal(0, 0, size=corners.shape)

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
            'img_path': density_path,
        }

        return self.process_data(raw_data)

    def process_data(self, data):
        img = data['image']
        corners = data['corners']
        annot = data['annot']

        # pre-process the image to use ImageNet-pretrained backbones
        img = img.transpose((2, 0, 1))
        raw_img = img.copy()
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = img.astype(np.float32)

        corners = np.array(corners)

        all_data = {
            "annot": annot,
            "name": data['name'],
            'img': img,
            'annot_path': data['annot_path'],
            'img_path': data['img_path'],
            'raw_img': raw_img,
        }

        # corner labels
        if not self.inference:
            pixel_labels, gauss_labels = self.get_corner_labels(corners)
            all_data['pixel_labels'] = pixel_labels
            all_data['gauss_labels'] = gauss_labels

        return all_data

    def random_aug_annot(self, img, annot, det_corners=None):
        # do random flipping
        img, annot, det_corners = self.random_flip(img, annot, det_corners)
        # return img, annot, None

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
            return img, annot, None

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

        return aug_img, aug_annot, None






