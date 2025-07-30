import torch
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleShapeDataset(Dataset):
    def __init__(self, num_samples=100, image_size=64):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = self.create_shape_dataset(image_size=self.image_size)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        label = np.random.randint(0, 3)
        label = torch.tensor(label).long()
        return image, label

    def create_shape_dataset(self, image_size=64):

        img = np.zeros((image_size, image_size), dtype=np.uint8)

        shape_type = np.random.choice([0, 1, 2])  # 0: Square, 1: Circle, 2: Triangle

        # size = np.random.randint(image_size // 8, image_size // 3)
        size = image_size // 4
        center = (image_size // 2, image_size // 2)

        if shape_type == 0:  # Square
            top_left = (center[0] - size, center[1] - size)
            bottom_right = (center[0] + size, center[1] + size)
            cv2.rectangle(img, top_left, bottom_right, color=255, thickness=-1)

        elif shape_type == 1:  # Circle
            radius = size
            cv2.circle(img, center, radius, color=255, thickness=-1)

        elif shape_type == 2:  # Triangle
            pt1 = (center[0], center[1] - size)
            pt2 = (center[0] - size, center[1] + size)
            pt3 = (center[0] + size, center[1] + size)
            pts = np.array([pt1, pt2, pt3], np.int32)
            cv2.fillPoly(img, [pts], color=255)

        return img


if __name__ == "__main__":
    import time
    dataset = SimpleShapeDataset(num_samples=1000, image_size=64)
    for i in range(10):
        image = dataset.create_shape_dataset()
        cv2.imshow("image", image)
        cv2.waitKey(0)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=4)
    tik = time.time()
    for images, masks in dataloader:
        
        print(images.shape)
        print(time.time() - tik)
        tik = time.time()
        
        