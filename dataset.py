# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the CC-BY-NC 
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at 
#
#     https://github.com/ByteDance-Seed/DeepFlow/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 


"""
DeepFlow Script for DataLoader.
"""


import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    """DataLoader for DeepFlow.


    DeepFlow's dataloader can load raw images, pre-extracted VAE features, and labels. 
    When "ssl_align" [1] is disabled, since training does not require raw images,
    it loads only the pre-extracted VAE features and labels.

    [1] Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think, 
        ICLR 2025
        Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie.
    """
    def __init__(self, data_dir, ssl_align):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.features_dir = os.path.join(data_dir, 'vae-sd')

        # images
        self.ssl_align = ssl_align
        if ssl_align:
            self.ssl_align = ssl_align
            self.images_dir = os.path.join(data_dir, 'images')
            self._image_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self.images_dir)
                for root, _dirs, files in os.walk(self.images_dir) for fname in files
                }
            self.image_fnames = sorted(
                fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
                )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        if self.ssl_align:
            assert len(self.image_fnames) == len(self.feature_fnames), \
                "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        feature_fname = self.feature_fnames[idx]
        features = np.load(os.path.join(self.features_dir, feature_fname))
        if self.ssl_align:
            image_fname = self.image_fnames[idx]
            image_ext = self._file_ext(image_fname)
            with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
                if image_ext == '.npy':
                    image = np.load(f)
                    image = image.reshape(-1, *image.shape[-2:])
                elif image_ext == '.png' and pyspng is not None:
                    image = pyspng.load(f.read())
                    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
                else:
                    image = np.array(PIL.Image.open(f))
                    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])
        else:
            return torch.from_numpy(features), torch.from_numpy(features), torch.tensor(self.labels[idx])


