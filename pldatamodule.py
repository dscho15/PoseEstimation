from pytorch_lightning import LightningDataModule
import albumentations as A
from dataset import ICBINDataset
from torch.utils.data import DataLoader, random_split

class ICBINDataModule(LightningDataModule):

    def __init__(self, 
                 transform: A.Compose = None,
                 path_to_meshes: str = "./datasets/meshes",
                 path_to_scenes: str = "./datasets/icbin"):

        super().__init__()
        
        self.path_to_meshes = path_to_meshes
        self.path_to_scenes = path_to_scenes
        self.transform = A.Compose([A.Normalize()]) if transform is None else transform
        self.prepare_data()
        self.setup()

    def prepare_data(self):

        self.icbin_dataset = ICBINDataset(path_to_scenes=self.path_to_scenes,
                                          path_to_meshes=self.path_to_meshes,
                                          transform=self.transform)
        
        self.train_size = int(0.8 * len(self.icbin_dataset))

        self.val_size = len(self.icbin_dataset) - self.train_size
        
    def setup(self):
        self.mnist_train, self.mnist_val = random_split(self.icbin_dataset, [self.train_size, self.val_size])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=16, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=16, num_workers=8, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return NotImplementedError

    def predict_dataloader(self):
        return NotImplementedError