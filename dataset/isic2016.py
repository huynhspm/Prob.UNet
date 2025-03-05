import cv2
import glob
import imageio
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from transform import TransformDataset

def get_isic2016_dataset(args, mode: str = ""):
    if mode == "train":
        train_dataset = ISIC2016Dataset(data_dir=args.data_path, train_val_test_dir='Train')
        train_dataset = TransformDataset(train_dataset, image_size=args.image_size)
        return train_dataset

    if mode == "val":
        val_dataset = ISIC2016Dataset(data_dir=args.data_path, train_val_test_dir='Test')
        val_dataset = TransformDataset(val_dataset, image_size=args.image_size)
        return val_dataset
    
    return train_dataset, val_dataset

class ISIC2016Dataset(Dataset):
    dataset_url = "https://challenge.isic-archive.com/data/#2016"
    train_test_mapping = {
        "Train": "ISBI2016_ISIC_Part3B_Training_Data",
        "Test": "ISBI2016_ISIC_Part3B_Test_Data"
    }
    
    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
    ) -> None:
        super().__init__()

        self.dataset_dir = data_dir

        if train_val_test_dir:
            train_val_test_dir = self.train_test_mapping[train_val_test_dir]
            self.dataset_dir = osp.join(self.dataset_dir, train_val_test_dir)
            self.img_paths = glob.glob(f"{self.dataset_dir}/*.jpg")
        else:
            self.img_paths = glob.glob(f"{self.dataset_dir}/*.jpg") if train_val_test_dir else [
                path for pattern in [
                    f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Test_Data/*.jpg",
                    f"{self.dataset_dir}/ISBI2016_ISIC_Part3B_Training_Data/*.jpg"
                ] for path in glob.glob(pattern)
            ]

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace(".jpg", "_Segmentation.png")

        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)
        mask = np.where(mask == 255, 1, 0)

        return mask, image


if __name__ == "__main__":
    dataset = ISIC2016Dataset(data_dir='../data/isic2016', train_val_test_dir="Train")
    print(len(dataset))

    mask, image = dataset[5]

    print(image.shape, image.dtype, type(image))
    print(mask.shape, mask.dtype, type(mask))

    print(np.unique(mask))

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()

    print(mask.max())