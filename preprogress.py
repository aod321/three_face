import torch
import torch.nn
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
import numpy as np
from PIL import Image


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
