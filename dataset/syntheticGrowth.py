import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Custom Dataset class for the synthetic growth dataset
class SyntheticGrowthDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # Load the dataset from the .npz file
        data = np.load(data_path)
        self.images = data['images']  # Shape: (num_samples, num_frames, image_size, image_size, 3)
        self.labels = data['labels']  # Shape: (num_samples,)

        # Convert to PyTorch tensors
        self.images = torch.from_numpy(self.images).permute(0, 1, 4, 2, 3).float() / 255.0
        # Shape: (num_samples, num_frames, 3, image_size, image_size)
        self.labels = torch.from_numpy(self.labels).long()

        # Normalize the images to [-1, 1]
        self.images = self.images * 2 - 1

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Get a single sample (sequence of images and label)
        images = self.images[idx]  # Shape: (num_frames, 3, image_size, image_size)
        label = self.labels[idx]   # Shape: ()
        return images, label
    
class SyntheticGrowthDataLoader(DataLoader):
    def __init__(self, data_path, batch_size=64, num_workers=4):
        
        # Initialize the datasets
        self.train_set = SyntheticGrowthDataset(f"{data_path}/train.npz")
        self.val_set = SyntheticGrowthDataset(f"{data_path}/val.npz")
        self.test_set = SyntheticGrowthDataset(f"{data_path}/test.npz")

        # Initialize the DataLoaders
        self.train_loader = DataLoader(
            self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        # Custom collate function to stack images and labels
        images, labels = zip(*batch)

        # First frame is the condition frame, label is the prompt, rest are the target frames
        labels = torch.stack(labels)
        condition_frame = torch.stack([img[0] for img in images])
        target_frames = torch.stack([img[1:] for img in images])

        # Reshape target_frames to (bs, num_frames * RGB, image_size, image_size)
        target_frames = target_frames.reshape(-1, target_frames.shape[1] * 3, target_frames.shape[3], target_frames.shape[3])

        return {
            "conditions": condition_frame,
            "images": target_frames,
            "prompt": labels
        }
    
    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader