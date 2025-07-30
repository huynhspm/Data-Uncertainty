import nibabel as nib
import os
import numpy as np
from pathlib import Path


def read_data(src_dir: str, des_dir: str):

    cases = os.listdir(src_dir)
    for case in cases:
        print(f"Processing: {case}")
        
        src_case_dir = src_dir / case
        des_case_dir = des_dir / case
        
        if not src_case_dir.is_dir():
            continue
        
        image = nib.load(src_case_dir / "image.nii.gz").get_fdata()
        mask1 = nib.load(src_case_dir / "task01_seg01.nii.gz").get_fdata()
        mask2 = nib.load(src_case_dir / "task01_seg02.nii.gz").get_fdata()
        
        assert image.shape == mask1.shape == mask2.shape, f"Mismatch at {sample_dir}"
        
        des_case_dir.mkdir(parents=True, exist_ok=True)
        for i in range(image.shape[0]):
            img_slice = image[i,:, :]
            m1_slice = mask1[i,:, :]
            m2_slice = mask2[i,:, :]
            np.save(des_case_dir / f"image_{i}.npy", img_slice)
            label = np.stack([m1_slice, m2_slice], axis=-1)
            np.save(des_case_dir / f"label_{i}.npy", label)

if __name__ == '__main__':
    data_dir = Path("../data")

    print("Processing QUBIQ pancreatic-lesion dataset...")
    read_data(
        src_dir=data_dir / "qubiq/training_data_v3_QC/pancreatic-lesion",
        des_dir=data_dir / "qubiq/pan_les/Train"
    )
    read_data(
        src_dir=data_dir / "qubiq/validation_data_qubiq2021_QC/pancreatic-lesion/Validation",
        des_dir=data_dir / "qubiq/pan_les/Val"
    )

    print("Processing QUBIQ pancreatic dataset...")
    read_data(
        src_dir=data_dir / "qubiq/training_data_v3_QC/pancreas",
        des_dir=data_dir / "qubiq/pan/Train"
    )
    read_data(
        src_dir=data_dir / "qubiq/validation_data_qubiq2021_QC/pancreas/Validation",
        des_dir=data_dir / "qubiq/pan/Val"
    )
