import os

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.limit = limit
        self.file_list = os.listdir(root_dir)[:limit]

    def __len__(self):
        if self.limit is not None:
            return min(len(self.file_list), self.limit)
        else:
            return len(self.file_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.file_list[item])
        img = Image.open(img_name)

        if self.transform:
            img = self.transform(img)

        return img


class ConvertToRGB(object):
    def __call__(self, img, *args, **kwargs):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
