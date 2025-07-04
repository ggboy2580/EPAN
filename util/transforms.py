from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import numpy as np

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img:Image.Image):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

if __name__ == '__main__':
    img=Image.open(r"D:\BaiduSyncdisk\01projs\02github_small\AlignedReID-master\util\aaaa.jpg")
    #transform = Random2DTranslation(256,128,0.5)
    import torchvision.transforms as transforms
    transform=transforms.Compose(
        [Random2DTranslation(256,128,0.5),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor()
         ]
    )
    img_t=transform(img)
    # import matplotlib.pyplot as plt
    # plt.figure(12)
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img_t)
    # plt.show()
    print(img_t)
    print(img_t.shape)

    pass