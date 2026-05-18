import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def crop_black_borders(img, threshold=20):
    arr = np.array(img)
    mask = arr.mean(axis=2) > threshold
    if not np.any(mask):
        return img
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return img.crop((x_min, y_min, x_max + 1, y_max + 1))

class H5GeologyDataset(Dataset):
    def __init__(self, h5_path, indices, label_map, transform=None):
        self.h5_path = h5_path
        self.indices = np.array(indices, dtype=np.int64)
        self.label_map = label_map
        self.transform = transform
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def _lazy_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def __getitem__(self, i):
        self._lazy_open()
        ridx = int(self.indices[i])

        img = self.h5_file["images"][ridx]
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        pil = Image.fromarray(img).convert("RGB")
        pil = crop_black_borders(pil)

        if self.transform:
            image = self.transform(pil)
        else:
            image = transforms.ToTensor()(pil)

        xrf = torch.tensor(self.h5_file["xrf"][ridx], dtype=torch.float32)
        y   = torch.tensor(self.h5_file["physical"][ridx], dtype=torch.float32)

        raw_label = self.h5_file["labels"][ridx]
        if isinstance(raw_label, bytes):
            raw_label = raw_label.decode('utf-8')
        else:
            raw_label = str(raw_label)

        label_idx = self.label_map.get(raw_label, -1)
        label = torch.tensor(label_idx, dtype=torch.long)

        return image, xrf, y, label

    def close(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass
            self.h5_file = None

def get_img_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
