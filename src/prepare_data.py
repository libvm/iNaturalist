from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label


def save_datasets(dataset, indices, output_dir):
    """
    Сохраняет изображения из датасета с применением transform в соответствующую папку.
    """
    os.makedirs(output_dir, exist_ok=True)
    for class_name in dataset.classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    for idx in indices:
        img_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        class_name = dataset.classes[label]
        dest_dir = os.path.join(output_dir, class_name)
        dest_path = os.path.join(dest_dir, os.path.basename(img_path))

        image = Image.open(img_path).convert("RGB")
        image.save(dest_path)


def prepare_data():
    data_dir = "images"
    dataset = MyDataset(root_dir=data_dir)

    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=dataset.labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=[dataset.labels[i] for i in temp_idx],
        random_state=42,
    )

    # Сохранение данных
    save_datasets(dataset, train_idx, "data_split/train")
    save_datasets(dataset, val_idx, "data_split/val")
    save_datasets(dataset, test_idx, "data_split/test")


if __name__ == "__main__":
    prepare_data()
