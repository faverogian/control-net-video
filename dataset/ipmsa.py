import os
import torch
from glob import glob
import lz4.frame
import numpy as np
import pickle
import torchvision.transforms as transforms
import torch.nn.functional as F
import ants
import time

class MRIImageKeys:
    '''
    A class to store the keys of the MRI images.
    '''
    FLAIR = 'FLAIR'
    GAD = 'GAD'
    CT2F = 'CT2F'
    NEWT2 = 'NEWT2'
    MASK = 'MASK'
    CLINICAL = 'CLINICAL'
    BRAIN_VOL = 'BRAIN'

class ClinicalKeys:
    '''
    A class to store the keys of the clinical data.
    '''
    AGE = 'AGE'
    EDSS = 'EDSS'
    TRIAL_ARM = 'TRIAL_ARM'
    SEX = 'SEX'
    GAD_COUNT = 'LESION_GAD_CONSENSUS_COUNT'
    T2_VOL = 'LESION_T2_VOL'


def get_leaf_val_from_dict(d: dict, keys):
    if len(keys) > 1:
        return get_leaf_val_from_dict(d[keys[0]], keys[1:])
    elif len(keys) == 1:
        if isinstance(d[keys[0]], dict):
            raise Exception("Not enough keys to reach leaf node")
        else:
            return d[keys[0]]
    else:
        raise Exception("Cannot have no keys")

def load_image(path: str):
    if path.endswith('.npy.lz4'):
        with lz4.frame.open(path, 'rb') as f:
            return np.load(f)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise Exception("File extension not supported!")

def glob_file(filepath_no_ext: str)->str:
    """
        filepath_no_ext: filepath with no file extension
        return: filepath with extension
        raises: Exception if no file found or more than one file found
    """
    files = glob(f"{filepath_no_ext}*")
    if len(files) == 0:
        raise Exception(f"No file starting with {filepath_no_ext} was found")
    if len(files) > 1:
        raise Exception(f"More than 1 file starting with {filepath_no_ext} was found")
    return files[0]

class IPMSADataset(torch.utils.data.Dataset):
    def __init__(self, VolumeLoaderPath: str, slurm: bool=False, num_samples: int=None):

        self.VolumeLoaderPath = VolumeLoaderPath

        with open(VolumeLoaderPath, 'rb') as f:
            dictionary = pickle.load(f)

        if slurm:
            self.rootdir_dict = {'MRI_AND_LABEL': os.environ['TMPDIR'], 'CLINICAL': os.environ['TMPDIR']}
        else:   
            self.rootdir_dict = dictionary['rootdir_dict']
        self.dataset_dict = dictionary['dataset_dict']
        self.item_template = dictionary['item_template']
        self.sample_keys = list(self.dataset_dict.keys())

        if num_samples is not None:
            self.sample_keys = self.sample_keys[:num_samples]

        self.transform = None

        self._validate_inputs()

    def _validate_inputs(self):
        # Check if rootdir_dict has paths
        for rootdir in self.rootdir_dict.values():
            assert os.path.isdir(rootdir), f"{rootdir} is not a valid directory"

        # Check if keys in each sample_dict can be found rootdir_dict
        for sample_key, sample_dict in self.dataset_dict.items():
            for key in sample_dict:
                assert key in self.rootdir_dict, f"{key} not found in rootdir_dict"

        # Check if first key in each keys_for_ch is in rootdir_dict
        for item_key, keys_for_ch_list in self.item_template.items():
            for keys_for_ch in keys_for_ch_list:
                assert len(keys_for_ch) > 0, "Cannot have empty keys"
                assert keys_for_ch[0] in self.rootdir_dict, f"First key for {keys_for_ch} must be in rootdir_dict"

        # Check if every path points to an actual file
        for sample_key, sample_dict in self.dataset_dict.items():
            for item_key, keys_for_ch_list in self.item_template.items():
                for keys_for_ch in keys_for_ch_list:
                    if keys_for_ch[0] == 'CLINICAL':
                        continue
                    try:
                        rel_path = get_leaf_val_from_dict(sample_dict, keys_for_ch)
                        full_path = os.path.join(self.rootdir_dict[keys_for_ch[0]], rel_path)
                        full_path = glob_file(full_path)
                        assert os.path.isfile(full_path)
                    except KeyError:
                        pass#print(f"Cannot find volume {keys_for_ch} for sample {sample_key}. Skipping.")

    def __len__(self):
        return len(self.sample_keys)
    
    def set_transform(self, transform):
        self.transform = transform

    def _load_vol(self, idx):
        sample_key = self.sample_keys[idx]
        sample_dict = self.dataset_dict[sample_key]

        # Load images
        output = {}
        filepaths = {}  # Store filepaths for each item_key
        for item_key, keys_for_ch_list in self.item_template.items():
            if item_key in ClinicalKeys.__dict__.values():
                tp_vals = []
                for keys_for_ch in keys_for_ch_list:
                    tp_vals.append(get_leaf_val_from_dict(sample_dict, keys_for_ch))
                output[item_key] = tp_vals
                filepaths[item_key] = None  # No filepath for clinical data
                continue
            # Get image paths
            img_paths = []
            for keys_for_ch in keys_for_ch_list:
                try:
                    rel_path = get_leaf_val_from_dict(sample_dict, keys_for_ch)
                    full_path = os.path.join(self.rootdir_dict[keys_for_ch[0]], rel_path)
                    full_path = glob_file(full_path)
                    img_paths.append(full_path)
                except KeyError:
                    pass
            
            # Load images and stack into one array
            imgs = [load_image(path) for path in img_paths]
            output[item_key] = np.stack(imgs)
            filepaths[item_key] = img_paths  # Store filepaths

        return output, filepaths
    
    def __getitem__(self, idx):
        output, filepaths = self._load_vol(idx)

        # Get trial and patient ID from first filepath
        filepaths = filepaths[list(filepaths.keys())[0]]
        trial_id = filepaths[0].split('/')[3]
        patient_id = filepaths[0].split('/')[4]
        idx = {'trial_id': trial_id, 'patient_id': patient_id}
        
        if self.transform is not None:
            output = self.transform({'output': output, 'idx': idx})

        return output
    
    def remove_condition(self, block_list):
        """
        Removes samples that do not meet the condition specified by condition_func.

        Args:
            condition_func (callable): Function that takes a sample (output of _load_vol) 
            as input and returns True if the sample is active, False otherwise.
        """
        # Open block list .txt file
        with open(block_list, 'r') as f:
            block_list = f.readlines()
        block_list = [line.strip() for line in block_list]
        block_list = [line.split('/') for line in block_list]

        # Get trial and patient IDs
        block_list = [{'trial_id': line[1], 'patient_id': line[2]} for line in block_list]

        inactive_idxs = []
        for idx in range(len(self)):
            output, filepaths = self._load_vol(idx)

            filepaths = filepaths[list(filepaths.keys())[0]]
            trial_id = filepaths[0].split('/')[3]
            patient_id = filepaths[0].split('/')[4]
            
            if {'trial_id': trial_id, 'patient_id': patient_id} in block_list:
                print(f"Sample {idx} is inactive")
                inactive_idxs.append(idx)

        # Remove inactive samples
        self.sample_keys = [key for i, key in enumerate(self.sample_keys) if i not in inactive_idxs]
        self.dataset_dict = {key: val for i, (key, val) in enumerate(self.dataset_dict.items()) if i not in inactive_idxs}

        # Save updated dataset to a new pickle file
        output_path = self.VolumeLoaderPath.replace('.pkl', '_filtered.pkl')
        data = {
            'rootdir_dict': self.rootdir_dict,
            'dataset_dict': self.dataset_dict,
            'item_template': self.item_template,
            'sample_keys': self.sample_keys
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
#----------------------------------------------------------------------------
# Transforms for MRI images

# Add the trial arm as a condition
trial_mapping = {
    'placebo': 0,
    'laquinimod': 0,
    'interferon beta-1a': 1,
    'dimethyl fumarate': 1,
    'ocrelizumab': 2,
}
inverse_trial_mapping = {
    0: 'NE',
    1: 'ME',
    2: 'HE',
}


class LORISTransforms:
    '''
    A series of custom transforms for preprocessing the MRI images for this experiment.
    '''

    class PadTimepoints:
        '''
        Ensures that the timepoints are padded to the maximum number of timepoints.
        For example, if two timepoints are needed as input and one of the MRI images
        used is a New T2 image, there will be only one timepoint available (the latter).
        This transform will pad the first timepoint of the New T2 image with zeros.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys 
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): An identical dictionary with the padded timepoints.
        '''
        def __call__(self, MRI_image):
            # Get maximum number of timepoints
            max_timepoints = max([MRI_image[key].shape[0] for key in MRI_image.keys() if key not in ClinicalKeys.__dict__.values()])
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue
                # Pad to maximum number of timepoints
                pad = max_timepoints - MRI_image[key].shape[0]
                MRI_image[key] = np.pad(MRI_image[key], ((pad,0), (0,0), (0,0), (0,0)), mode='constant', constant_values=0)
            return MRI_image

    class GetSlice:
        '''
        Extracts a slice(s) from the MRI image. If the slice is a single slice,
        only the middle slice is extracted. If multiple slices are requested, the
        slices are extracted from the middle of the MRI image.
        e.g., if 3 slices are requested, the slices extracted are [center-1, center, center+1].

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys 
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): An identical dictionary with the requested slices.
        '''
        def __init__(self, slices):
            assert slices % 2 != 0, "Number of slices must be odd!"
            self.slices = slices // 2
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue
                # MRI image shape [t, D, H, W]
                center = MRI_image[key].shape[1] // 2
                if self.slices == 0:
                    MRI_image[key] = MRI_image[key][:, center, :, :]
                    MRI_image[key] = np.expand_dims(MRI_image[key], axis=-3)
                else:
                    MRI_image[key] = MRI_image[key][:, center-self.slices:center+self.slices+1, :, :]
            return MRI_image

    class Denoise:
        '''
        Denoises the MRI image using a BEAST mask.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] and the mask ['MASK'].

        Returns:
            MRI_image (dict): A dictionary containing the denoised MRI image 
            with the 'MASK' key removed.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue
                MRI_image[key] = MRI_image[key] * MRI_image[MRIImageKeys.MASK] 
            return MRI_image

    class BinarizeLabel:
        '''
        Binarizes the label images to 0 and 1. Label images are ['CT2F'], ['NEWT2'], and ['GAD'].

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the MRI image with binarized labels.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in [MRIImageKeys.CT2F, MRIImageKeys.NEWT2, MRIImageKeys.GAD]:
                    MRI_image[key] = (MRI_image[key] > 0).astype(np.float32)
            return MRI_image

    class Resize:
        '''
        Resizes the MRI image to next power of 2 in both dimensions. 
        Assumes that the image has dimensions (t, D, H, W).

        Args:
            MRI_image (dict): A dictionary containing the MRI images with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL']. 

        Returns:
            MRI_image (dict): A dictionary containing the resized MRI images.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue
                w, h = MRI_image[key].shape[2], MRI_image[key].shape[3]
                max_dim = max(w, h)
                next_power_of_2 = 2 ** ((max_dim - 1).bit_length())
                pad_w = (next_power_of_2 - w)
                pad_h = (next_power_of_2 - h)
                MRI_image[key] = np.pad(MRI_image[key], ((0,0), (0,0), (pad_w//2,pad_w//2), (pad_h//2,pad_h//2)), mode='minimum')
            return MRI_image

    class Normalize:
        '''
        Normalizes the MRI image between [-1, 1].

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the normalized MRI images.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue

                MRI = MRI_image[key]

                # Get (min,max) range from the reference image (first FLAIR timepoint)
                if key == MRIImageKeys.FLAIR:
                    # Clip the MRI image to 99th percentile
                    mean = np.mean(MRI, axis=(-2,-1), keepdims=True)
                    std = np.std(MRI, axis=(-2,-1), keepdims=True)
                    MRI = np.clip(MRI, mean - 4 * std, mean + 4 * std)

                    reference_MRI = MRI[0]
                    reference_max = reference_MRI.max()
                    reference_min = reference_MRI.min()

                    # Normalize based on reference max and min
                    MRI = (MRI - reference_min) / (reference_max - reference_min + 1e-12) # [0, 1]
                    MRI = np.clip(MRI, 0, 1)

                MRI = (MRI - 0.5) / 0.5 # [-1, 1]
                MRI_image[key] = MRI

            return MRI_image
        
    class NormalizeTensor:
        '''
        Normalizes the MRI image between [-1, 1].

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the normalized MRI images.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue

                MRI = MRI_image[key]

                # Ensure tensor is on the correct device (CPU or GPU)
                device = MRI.device

                # Get (min, max) range from the reference image (first FLAIR timepoint)
                if key == MRIImageKeys.FLAIR:
                    # Clip the MRI image to 99th percentile
                    mean = MRI.mean(dim=(-2, -1), keepdim=True)
                    std = MRI.std(dim=(-2, -1), keepdim=True)
                    MRI = torch.clamp(MRI, mean - 4 * std, mean + 4 * std)

                    reference_MRI = MRI[0]
                    reference_max = reference_MRI.max()
                    reference_min = reference_MRI.min()

                    # Normalize based on reference max and min
                    MRI = (MRI - reference_min) / (reference_max - reference_min + 1e-12)  # [0, 1]
                    MRI = torch.clamp(MRI, 0, 1)

                # Normalize to [-1, 1]
                MRI = (MRI - 0.5) / 0.5  # [-1, 1]
                MRI_image[key] = MRI

            return MRI_image
            
    class SegmentBrain:
        '''
        Segments the (GM+WM, CSF) regions from the FLAIR image using a threshold.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing one extra key ['BRAIN'].
        '''
        def __call__(self, MRI_image):
            segmentations = []
            for key in MRI_image.keys():
                if key != MRIImageKeys.FLAIR:
                    continue

                MRI = MRI_image[key] / 2 + 0.5 # [0, 1]

                for image in MRI:
                    image = image.squeeze()
                    ants_image = ants.from_numpy(image)
                    mask = ants.get_mask(ants_image)
                    segmentation = ants.atropos(a=ants_image, m='[0.8,1x1]', c='[1,0]', i='kmeans[2]', x=mask)
                    segmentation = segmentation["segmentation"]

                    # Convert to numpy array
                    segmentation = segmentation.numpy()

                    # Add segmentation to list
                    segmentations.append(segmentation)

            segmentation = np.stack(segmentations)

            # Add segmentation to dict
            MRI_image['BRAIN'] = segmentation

            return MRI_image

    class BlurLabel2D:
        '''
        Blurs the New T2 label image using a Gaussian filter.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the blurred New T2 label image.
        '''
        def __init__(self, depth=1):
            self.depth = depth

        def __call__(self, MRI_image):
            # Create a 5x5 Gaussian kernel
            kernel = torch.tensor([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])

            # Create the xy Gaussian kernel
            kernel_xy = kernel.unsqueeze(0)
            kernel_xy = torch.cat([kernel_xy for _ in range(3)], dim=0)
            
            # Create the z Gaussian kernel of length depth
            kernel_z = torch.linspace(-(self.depth // 2), self.depth // 2, self.depth)
            kernel_z = torch.exp(-kernel_z ** 2 / (2 * 1 ** 2))

            # Create the 3D Gaussian kernel
            kernel_3d = torch.stack([kernel_xy[i, :, :] * scale for i, scale in enumerate(kernel_z)], dim=0)
            kernel_3d = kernel_3d / kernel_3d.sum()
            kernel_3d = kernel_3d.to(torch.float32)
            kernel_3d = kernel_3d.unsqueeze(0)

            # Blur the New T2 image
            MRI_image[MRIImageKeys.NEWT2] = F.pad(MRI_image[MRIImageKeys.NEWT2], (2, 2, 2, 2), mode='constant', value=-1)
            MRI_image[MRIImageKeys.NEWT2] = F.conv2d(input=MRI_image[MRIImageKeys.NEWT2], weight=kernel_3d)

            # Blur the CT2F image
            MRI_image[MRIImageKeys.CT2F] = F.pad(MRI_image[MRIImageKeys.CT2F], (2, 2, 2, 2), mode='constant', value=-1)
            MRI_image[MRIImageKeys.CT2F] = F.conv2d(input=MRI_image[MRIImageKeys.CT2F], weight=kernel_3d)

            return MRI_image
        
    class BlurLabel3D:
        '''
        Blurs the New T2 label image using a Gaussian filter.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the blurred New T2 label image.
        '''
        def __init__(self, sigma=1, kernel_size=5):
            self.sigma = sigma
            self.kernel_size = kernel_size

        def __call__(self, MRI_image):
            def gaussian_kernel_3d(sigma, kernel_size=3):
                # Calculate the radius of the kernel
                radius = (kernel_size - 1) // 2

                # Create a grid of coordinates
                grid = torch.stack(torch.meshgrid([
                    torch.arange(-radius, radius + 1),
                    torch.arange(-radius, radius + 1),
                    torch.arange(-radius, radius + 1)
                ], indexing='ij'))

                # Calculate the squared distances from the center
                squared_distances = (grid ** 2).sum(dim=0)

                # Calculate the Gaussian values
                kernel = torch.exp(-squared_distances / (2 * sigma ** 2))

                # Normalize the kernel
                kernel /= kernel.sum()

                return kernel
            
            kernel_3d = gaussian_kernel_3d(sigma=self.sigma, kernel_size=self.kernel_size).unsqueeze(0).unsqueeze(0)

            # Blur the New T2 image
            MRI_image[MRIImageKeys.NEWT2] = F.conv3d(MRI_image[MRIImageKeys.NEWT2].unsqueeze(1), kernel_3d, padding=2)
            MRI_image[MRIImageKeys.NEWT2] = MRI_image[MRIImageKeys.NEWT2].squeeze(1)

            # Blur the CT2F image
            MRI_image[MRIImageKeys.CT2F] = F.conv3d(MRI_image[MRIImageKeys.CT2F].unsqueeze(1), kernel_3d, padding=2)
            MRI_image[MRIImageKeys.CT2F] = MRI_image[MRIImageKeys.CT2F].squeeze(1)

            return MRI_image

    class ToTensor:
        '''
        Converts the MRI image to a PyTorch tensor.

        Args:
            MRI_image (dict): A dictionary containing the MRI image with keys
            ['FLAIR'], ['GAD'], ['CT2F'] ... ['MASK'] ... ['CLINICAL'].

        Returns:
            MRI_image (dict): A dictionary containing the PyTorch tensor MRI images.
        '''
        def __call__(self, MRI_image):
            for key in MRI_image.keys():
                if key in ClinicalKeys.__dict__.values():
                    continue
                MRI_image[key] = torch.tensor(MRI_image[key])
            return MRI_image
        
class IPMSADataLoader:
    def __init__(self, train_data_path, val_data_path, test_data_path, collate_fn, slurm=0, batch_size=64, num_workers=4):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Initialize datasets
        self.train_dataset = IPMSADataset(train_data_path, slurm=slurm)
        self.val_dataset = IPMSADataset(val_data_path, slurm=slurm)
        self.test_dataset = IPMSADataset(test_data_path, slurm=slurm)

        # Set the transform for the datasets
        self.train_dataset.set_transform(collate_fn)
        self.val_dataset.set_transform(collate_fn)
        self.test_dataset.set_transform(collate_fn)

        # Initialize DataLoaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_test_loader(self):
        return self.test_loader