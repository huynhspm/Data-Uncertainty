import os
import h5py
import numpy as np
from pathlib import Path

def read_data(src_dir: str, des_dir: str):
    h5_keys = {
        "image": ["t1", "t1c", "t2"],
        "label": ["label_a1", "label_a2", "label_a3", "label_a4"],
    }

    cases = os.listdir(src_dir)
    for case in cases:
        
        case_name = case.split(".")[0]
        case_dir = des_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = src_dir / case
        with h5py.File(file_path, "r") as file:
            for name, image_types  in h5_keys.items():
                images = []
                for image_type in image_types:
                    images.append(file[image_type])
                images = np.stack(images, axis=-1)
                for i in range(images.shape[0]):
                    np.save(case_dir / f"{name}_{i}.npy", images[i])

if __name__ == '__main__':
    data_dir = Path("../data/mmis")
    read_data(src_dir=data_dir / "MMIS2024TASK1/training", des_dir=data_dir / "Train")
    read_data(src_dir=data_dir / "MMIS2024TASK1/validation", des_dir=data_dir / "Val")