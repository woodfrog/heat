import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import cv2

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class CornersDataset(Dataset):
    def __init__(self, image_size=256, inference=False):
        super(CornersDataset, self).__init__()
        self.image_size = image_size
        self.inference = inference
        self._data_names = []

    def __len__(self):
        raise len(self._data_names)

    def __getitem__(self, idx):
        raise NotImplementedError

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
            'det_path': data['det_path'],
            'raw_img': raw_img,
        }

        # corner labels for training
        if not self.inference:
            pixel_labels, gauss_labels = self.get_corner_labels(corners)
            all_data['pixel_labels'] = pixel_labels
            all_data['gauss_labels'] = gauss_labels

        return all_data

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

        # Do the corner transform within a big matrix transformation
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
