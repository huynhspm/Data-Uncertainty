import os
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

def read_data(data_dir: str):
    data = {}
    max_bytes = 2**31 - 1
    for file in os.listdir(data_dir):
        filename = os.fsdecode(file)
    
    if '.pickle' in filename:
            print("Loading file", filename)
            file_path = os.path.join(data_dir, filename)
            bytes_in = bytearray(0)
            input_size = os.path.getsize(file_path)
            with open(file_path, 'rb') as f_in:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += f_in.read(max_bytes)
            new_data = pickle.loads(bytes_in)
            data.update(new_data)
    
    return data

def save_data(data, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

def split_data(data_dir: str, val_size=0.1, random_state=42):
    data = read_data(data_dir)
    grouped_data = defaultdict(list)
    for _, value in data.items():
        grouped_data[value["series_uid"]].append((value["image"], value["masks"]))
    
    all_uids = list(grouped_data.keys())
    train_uids, val_uids = train_test_split(all_uids, test_size=val_size, random_state=random_state)

    def create_subset(uids):
        images, labels = [], []
        for uid in uids:
            for img, label in grouped_data[uid]:
                images.append(img)
                labels.append(label)
        return images, labels

    train_images, train_labels = create_subset(train_uids)
    val_images, val_labels = create_subset(val_uids)

    print(len(train_images), len(train_labels), len(val_images), len(val_labels))
    
    save_data({"images": train_images, "masks": train_labels}, save_path=f"{data_dir}/Train.pickle")
    save_data({"images": val_images, "masks": val_labels}, save_path=f"{data_dir}/Val.pickle")

if __name__ == '__main__':
    data_dir = "../data/lidc/"
    split_data(data_dir)