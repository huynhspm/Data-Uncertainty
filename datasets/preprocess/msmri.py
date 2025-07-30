from typing import List
import os
import shutil
import random
import nibabel
import numpy as np
from pathlib import Path


def preprocess(src_dir: str, patient_dirs: List[str], des_dir: str, img_types: List[str], mask_types: List[str]):

    # create folder
    Path(des_dir).mkdir(parents=True, exist_ok=True)

    for patient_dir in patient_dirs:
        patient_src_dir = os.path.join(src_dir, patient_dir)
        if not os.path.isdir(patient_src_dir): continue

        # create folder
        patient_des_dir = os.path.join(des_dir, patient_dir)
        Path(patient_des_dir).mkdir(parents=True, exist_ok=True)
        
        pp_dir = os.path.join(patient_src_dir, "preprocessed")
        mask_dir = os.path.join(patient_src_dir, "masks")
        
        files = os.listdir(pp_dir)
        n_time_point = len(files) // len(img_types)
        
        for time_point in range(n_time_point):
            time_point_name = "{}_{:02d}".format(patient_dir, time_point + 1)
            
            # read images
            images = []
            for img_type in img_types:
                pp_file_path = os.path.join(pp_dir, f"{time_point_name}_{img_type}_pp.nii")
                img = np.array(nibabel.load(pp_file_path).get_fdata())
                images.append(img)
            images = np.stack(images, axis=2)
            
            # read mask
            masks = []
            for mask_type in mask_types:
                mask_file_path = os.path.join(mask_dir, f"{time_point_name}_{mask_type}.nii")
                mask = np.array(nibabel.load(mask_file_path).get_fdata())
                masks.append(mask)
            masks = np.stack(masks, axis=2)

            for slice_id in range(images.shape[-1]):
                image_name = f"{patient_des_dir}/time_point_{time_point}_image_slice_{slice_id}"
                mask_name = f"{patient_des_dir}/time_point_{time_point}_mask_slice_{slice_id}"
                if images[..., slice_id].sum() == 0: 
                    continue
                np.save(image_name, images[..., slice_id])
                np.save(mask_name, masks[..., slice_id])

if __name__ == "__main__":
    data_rul = None
    data_dir = "../data/msmri"
    src_dir = f"{data_dir}/training"
    img_types = ["flair", "mprage", "pd", "t2"]
    mask_types = ["mask1", "mask2"]
    
    preprocess(src_dir=src_dir,
            patient_dirs=["training01", "training02", "training03", "training04"],
            des_dir=f"{data_dir}/Train",
            img_types=img_types,
            mask_types=mask_types)
    
    preprocess(src_dir=src_dir,
            patient_dirs=["training05"],
            des_dir=f"{data_dir}/Val",
            img_types=img_types,
            mask_types=mask_types)