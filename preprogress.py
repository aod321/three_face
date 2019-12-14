import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from PIL import Image
from skimage.util import random_noise
from PIL import ImageFilter, Image


def resize_img_keep_ratio(img, target_size):
    old_size = img.shape[0:2]
    # ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])

    interpol = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR

    img = cv2.resize(img, dsize=(new_size[1], new_size[0]), interpolation=interpol)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        labels = np.uint8(labels)
        labels = TF.resize(TF.to_pil_image(labels), self.size, Image.ANTIALIAS)
        # image = TF.resize(TF.to_pil_image(image), self.size, Image.ANTIALIAS)
        resized_image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)

        sample = {'image': resized_image,
                  'labels': labels
                  }
        return sample


class Stage2Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.
             Override the __call__ of transforms.Resize
    """

    def __call__(self, sample):
        """
            Args:
                 sample:{'image':PIL Image to be resized,'labels':labels to be resized}

             Returns:
                 sample:{'image':resized PIL Image,'labels': resized PIL label list}

        """
        image, labels = sample['image'], sample['labels']
        resized_image = np.array([cv2.resize(image[i], self.size, interpolation=cv2.INTER_AREA)
                                  for i in range(len(image))])
        labels = {x: np.array([np.array(TF.resize(TF.to_pil_image(labels[x][r]), self.size, Image.ANTIALIAS))
                               for r in range(len(labels[x]))])
                  for x in ['eye1', 'eye2', 'nose', 'mouth']
                  }

        sample = {'image': resized_image,
                  'labels': labels
                  }

        return sample


class Stage2_ToPILImage(object):
    """Convert a  ``numpy.ndarray`` to ``PIL Image``

    """

    def __call__(self, sample):
        """
                Args:
                    dict of sample (numpy.ndarray): Image and Labels to be converted.

                Returns:
                    dict of sample(PIL,List of PIL): Converted image and Labels.
        """
        image, labels = sample['image'], sample['labels']
        image = [TF.to_pil_image(image[i])
                 for i in range(len(image))]
        labels = {x: [TF.to_pil_image(labels[x][i])
                      for i in range(len(labels[x]))]
                  for x in ['eye1', 'eye2', 'nose', 'mouth']
                  }

        return {'image': image,
                'labels': labels
                }

class Stage1ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image = sample['image']
        labels = sample['labels']
        image = TF.to_tensor(image)
        labels = TF.to_tensor(labels)

        return {'image': image,
                'labels': labels
                }


class ToTensor(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

         Override the __call__ of transforms.ToTensor
    """

    def __call__(self, sample):
        """
                Args:
                    dict of pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

                Returns:y
                    Tensor: Converted image.
        """
        image = sample['image']
        labels = sample['labels']
        image = torch.stack([TF.to_tensor(image[i])
                             for i in range(len(image))])

        labels = {x: torch.cat([TF.to_tensor(labels[x][r])
                                for r in range(len(labels[x]))
                                ])
                  for x in ['eye1', 'eye2', 'nose', 'mouth']
                  }

        return {'image': image,
                'labels': labels
                }


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.

        Override the __call__ of transforms.RandomRotation

    """

    def __call__(self, sample):
        """
            sample (dict of PIL Image and label): Image to be rotated.

        Returns:
            Rotated sample: dict of Rotated image.
        """

        angle = self.get_params(self.degrees)

        img, labels = sample['image'], sample['labels']

        rotated_img = [TF.rotate(img[i], angle, self.resample, self.expand, self.center)
                       for i in range(len(img))]
        rotated_labels = {x: [TF.rotate(labels[x][r], angle, self.resample, self.expand, self.center)
                              for r in range(len(labels[x]))
                              ]
                          for x in ['eye1', 'eye2', 'nose', 'mouth']
                          }

        sample = {'image': rotated_img,
                  'labels': rotated_labels
                  }

        return sample


class RandomResizedCrop(transforms.RandomResizedCrop):
    """Crop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.RandomResizedCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']

        croped_img = []
        for r in range(len(img)):
            i, j, h, w = self.get_params(img[r], self.scale, self.ratio)
            croped_img.append(TF.resized_crop(img[r], i, j, h, w, self.size, self.interpolation))


        croped_labels = {x: [TF.resized_crop(labels[x][r], i, j, h, w, self.size, self.interpolation)
                             for r in range(len(labels[x]))
                             ]
                         for x in ['eye1', 'eye2', 'nose', 'mouth']
                         }

        sample = {'image': croped_img,
                  'labels': croped_labels
                  }

        return sample


class HorizontalFlip(object):
    """ HorizontalFlip the given PIL Image
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be

        Returns:
        """
        img, labels = sample['image'], sample['labels']

        img = [TF.hflip(img[r])
               for r in range(len(img))]

        labels = {x: [TF.hflip(labels[r])
                  for r in range(len(labels))
                      ]
                  for x in ['eye1', 'eye2', 'nose', 'mouth']
                  }

        sample = {'image': img,
                  'labels': labels
                  }

        return sample


class CenterCrop(transforms.CenterCrop):
    """CenterCrop the given PIL Image to random size and aspect ratio.

        Override the __call__ of transforms.CenterCrop
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        img, labels = sample['image'], sample['labels']

        croped_img = [TF.center_crop(img[r], self.size)
                      for r in range(len(img))]
        croped_labels = {x: [TF.center_crop(labels[r], self.size)
                         for r in range(len(labels))
                             ]
                         for x in ['eye1', 'eye2', 'nose', 'mouth']
                         }

        sample = {'image': croped_img,
                  'labels': croped_labels
                  }

        return sample


class RandomAffine(transforms.RandomAffine):

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, labels = sample['image'], sample['labels']
        img_in = []
        for r in range(len(img)):
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img[r].size)
            img_in.append(TF.affine(img[r], *ret, resample=self.resample, fillcolor=self.fillcolor))

        labels = {x: [TF.affine(labels[x][r], *ret, resample=self.resample, fillcolor=self.fillcolor)
                  for r in range(len(labels[x]))]
                  for x in ['eye1', 'eye2', 'nose', 'mouth']
                  }
        sample = {'image': img,
                  'labels': labels
                  }
        return sample


class GaussianNoise(object):
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = [TF.to_pil_image(np.uint8(255 * random_noise(np.array(img[r], np.uint8))))
               for r in range(len(img))
               ]
        sample = {'image': img,
                  'labels': labels
                  }

        return sample


class Blurfilter(object):
    # img: PIL image
    def __call__(self, sample):
        img, labels = sample['image'], sample['labels']
        img = [img[r].filter(ImageFilter.BLUR)
               for r in range(len(img))]
        sample = {'image': img,
                  'labels': labels
                  }

        return sample


class LabelsToOneHot(object):
    """
        LabelsToOneHot
    """

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            one-hot numpy label:
        """
        img, labels = sample['image'], sample['labels']

        #  Use auto-threshold to binary the labels into one-hot
        new_labels = []
        for i in range(len(labels)):
            temp = np.array(labels[i], np.uint8)
            _, binary = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            new_labels.append(binary)

        sample = {'image': img,
                  'labels': new_labels
                  }

        return sample
