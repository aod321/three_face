from torch.utils.data import Dataset
import jpeg4py as jpeg
import os
import numpy as np
import torchvision
from skimage import io
import jpeg4py as jpeg
import torch


class PartsDataset(Dataset):
    # HelenDataset
    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.name_list = np.loadtxt(os.path.join(root_dir, txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.transform = transform
        self.names = ['eye1', 'eye2', 'nose', 'mouth']
        self.label_id = {'eye1': [2, 4],
                         'eye2': [3, 5],
                         'nose': [6],
                         'mouth': [7, 8, 9]
                         }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        part_path = [os.path.join(self.root_dir, '%s' % x, 'images',
                                  img_name + '.jpg')
                     for x in self.names]
        labels_path = {x: [os.path.join(self.root_dir, '%s' % x,
                                        'labels', img_name,
                                        img_name + "_lbl%.2d.png" % i)
                           for i in self.label_id[x]]
                       for x in self.names}

        parts_image = [io.imread(part_path[i])
                       for i in range(4)]

        labels = {x: np.array([io.imread(labels_path[x][i])
                               for i in range(len(self.label_id[x]))
                               ])
                  for x in self.names
                  }

        for x in self.names:
            bg = 255 - np.sum(labels[x], axis=0, keepdims=True)  # [1, 64, 64]
            labels[x] = np.uint8(np.concatenate([bg, labels[x]], axis=0))  # [L + 1, 64, 64]

        # labels = {'eye1':,
        #           'eye2':,
        #           'nose':,
        #           'mouth':}

        sample = {'image': parts_image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        return sample
