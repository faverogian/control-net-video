import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Map class to CIFAR10 class names
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10DataLoader:
    def __init__(self, data_path, batch_size=64, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations for the datasets
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Initialize datasets
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        return {
            "images": torch.stack(images),
            "prompt": torch.tensor(labels)
        }
    
    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader