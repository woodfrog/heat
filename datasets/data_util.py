from PIL import ImageFilter
from torchvision import transforms


def RandomBlur(radius=2.):
    blur = GaussianBlur(radius=radius)
    full_transform = transforms.RandomApply([blur], p=.3)
    return full_transform


class ImageFilterTransform(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        return img.filter(self.filter)


class GaussianBlur(ImageFilterTransform):

    def __init__(self, radius=2.):
        self.filter = ImageFilter.GaussianBlur(radius=radius)
