import glob
import imageio
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from transform import TransformDataset

def get_isic2018_dataset(args, mode: str = ""):
    if mode == "train":
        train_dataset = ISIC2018Dataset(data_dir=args.data_path, train_val_test_dir='Train')
        train_dataset = TransformDataset(train_dataset, image_size=args.image_size)
    
    if mode == "val":
        val_dataset = ISIC2018Dataset(data_dir=args.data_path, train_val_test_dir='Val')    
        val_dataset = TransformDataset(val_dataset, image_size=args.image_size)
    
    return train_dataset, val_dataset

class ISIC2018Dataset(Dataset):

    dataset_url = 'https://challenge.isic-archive.com/data/#2018'
    train_val_test_mapping = {
        "Train": "ISIC2018_Task1-2_Training_Input",
        "Val": "ISIC2018_Task1-2_Validation_Input",
        "Test": "ISIC2018_Task1-2_Test_Input"
    }
    def __init__(
        self,
        data_dir: str = 'data',
        train_val_test_dir: str = None,
    ) -> None:
        super().__init__()

        self.dataset_dir = data_dir
        if train_val_test_dir:
            train_val_test_dir = self.train_val_test_mapping[train_val_test_dir]
            self.dataset_dir = osp.join(self.dataset_dir, train_val_test_dir)
            self.img_paths = glob.glob(f"{self.dataset_dir}/*.jpg")
        else:
            img_dirs = [
                f"{self.dataset_dir}/ISIC2018_Task1-2_Test_Input/*.jpg",
                f"{self.dataset_dir}/ISIC2018_Task1-2_Training_Input/*.jpg",
                f"{self.dataset_dir}/ISIC2018_Task1-2_Validation_Input/*.jpg",
            ]

            self.img_paths = [
                img_path for img_dir in img_dirs
                for img_path in glob.glob(img_dir)
            ]

    def prepare_data(self) -> None:
        pass

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace('1-2', '1').replace(
            'Input', 'GroundTruth').replace('.jpg', '_segmentation.png')

        image = imageio.v2.imread(img_path)
        mask = imageio.v2.imread(mask_path)
        mask = np.where(mask == 255, 1, 0)

        return mask, image


if __name__ == "__main__":
    dataset = ISIC2018Dataset(data_dir='data/isic2018', train_val_test_dir=None)
    print(len(dataset))

    mask, cond = dataset[0]
    image = cond['image']

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