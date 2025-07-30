from typing import Literal
import os
import pickle
import numpy as np
import random
from torch.utils.data import Dataset


class LIDCDataset(Dataset):

    dataset_url = 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254'
    masks_per_image = 4

    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
        mask_type: Literal["ensemble", "random", "multi"] = "ensemble",
    ) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.mask_type = mask_type
        self.data = {"images": [], "masks": []}
        
        if train_val_test_dir:
            self.load_data(f"{self.data_dir}/{train_val_test_dir}.pickle")
        else:
            self.load_data(f"{self.data_dir}/Train.pickle")
            self.load_data(f"{self.data_dir}/Val.pickle")

    def load_data(self, datafile_path: str):
        max_bytes = 2**31 - 1
        
        # print("Loading file", datafile_path)
        bytes_in = bytearray(0)
        input_size = os.path.getsize(datafile_path)
        with open(datafile_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        self.data["images"].extend(new_data["images"])
        self.data["masks"].extend(new_data["masks"])

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index):
        image = self.data["images"][index]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        if self.mask_type == "random":
            mask_id = random.randint(0, self.masks_per_image - 1)
            mask = self.data["masks"][index][mask_id]
        else:
            mask = self.data["masks"][index]
            if self.mask_type == "ensemble":
                mask = np.stack(mask, axis=-1).mean(axis=-1)
                mask = (mask > 0.5).astype(np.uint8)
            elif self.mask_type == "multi":
                pass
        
        return mask, image


if __name__ == "__main__":
    dataset = LIDCDataset(data_dir='../data/lidc', mask_type="multi", train_val_test_dir="Train")
    print(len(dataset))

    id = random.randint(0, len(dataset) - 1)
    print(id)
    masks, image = dataset[id]
    print(image.shape, image.dtype, type(image))
    
    masks = np.stack(masks, axis=-1)
    mask_e = masks.mean(axis=-1)
    mask_var = masks.var(axis=-1)

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 0].axis("off")
    
    axs[0, 1].imshow(mask_e, cmap='gray')
    axs[0, 1].set_title('Mask_e')
    axs[0, 1].axis("off")
    
    axs[0, 2].imshow(masks[:, :, 0], cmap='gray')
    axs[0, 2].set_title('Mask_0')
    axs[0, 2].axis("off")
    
    axs[1, 0].imshow(masks[:, :, 1], cmap='gray')
    axs[1, 0].set_title('Mask_1')
    axs[1, 0].axis("off")
    
    axs[1, 1].imshow(masks[:, :, 2], cmap='gray')
    axs[1, 1].set_title('Mask_2')
    axs[1, 1].axis("off")
    
    axs[1, 2].imshow(masks[:, :, 3], cmap='gray')
    axs[1, 2].set_title('Mask_3')
    axs[1, 2].axis("off")
    
    axs[2, 0].imshow(mask_var, cmap='gray')
    axs[2, 0].set_title('Variance')
    axs[2, 0].axis("off")
    
    axs[2, 1].imshow(mask_var, cmap='gray')
    axs[2, 1].set_title('Variance')
    axs[2, 1].axis("off")
    
    axs[2, 2] = sns.heatmap(mask_var)
    axs[2, 2].set_title('Variance')
    axs[2, 2].axis("off")
    
    plt.tight_layout()
    plt.show()