# Project path
import sys
import os
import json

# Get project root from environment variable
projectroot = os.environ['PROJECT_ROOT']
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)
os.chdir(projectroot)

# control-net-video imports
from dataset.syntheticGrowth import SyntheticGrowthDataLoader
from nets.unet import UNetCondition2D, ControlUNet
from diffusion.control_net import ControlNet

# Third party imports
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import accelerate

# Training configuration
class TrainingConfig:
    def __init__(self):
        config_str = os.environ.get('TRAINING_CONFIG')
        if config_str is None:
            raise ValueError("TRAINING_CONFIG environment variable is not set")

        self.config = json.loads(config_str)
        self.project_root = self.config['project_root']
        self.experiment_dir = self.config['experiment_dir']

        # Construct experiment path
        self.experiment_path = os.path.join(f"{self.project_root}{self.experiment_dir}")

    def __getattr__(self, name):
        return self.config.get(name)
    
def plotter(output_dir, batches, samples, epoch, process_idx):
    """
    Plots all RGB frames of all samples in the provided dataset 
    and saves each sample's frames as an image to the specified directory.

    Args:
        output_dir (str): Directory to save the plotted images.
        batches (int): Number of batches (not used in this function but kept for consistency).
        samples (np.ndarray): Dataset samples, shape (num_samples, num_frames, height, width, channels).
        epoch (int): Current epoch number (used for the file names).
        process_idx (int): Process index (used for the file names).

    Returns:
        list: List of file paths to the saved images.
    """
    import matplotlib.pyplot as plt

    # Iterate over all samples
    for sample_idx, sample_batches in enumerate(samples):
        sample_batches = sample_batches.cpu().numpy() * 0.5 + 0.5 # Rescale to [0, 1]
        condition_frame = batches[sample_idx]['conditions'].cpu().numpy() * 0.5 + 0.5 # Rescale to [0, 1]
        target_frames = batches[sample_idx]['images'].cpu().numpy() * 0.5 + 0.5 # Rescale to [0, 1]
        label = batches[sample_idx]['prompt']

        # Reshape the images from (bs, num_frames * channels, height, width) to (bs, num_frames, height, width, channels)
        sample_batches = sample_batches.reshape(sample_batches.shape[0], sample_batches.shape[1] // 3, 3, sample_batches.shape[2], sample_batches.shape[3])
        target_frames = target_frames.reshape(target_frames.shape[0], target_frames.shape[1] // 3, 3, target_frames.shape[2], target_frames.shape[3])

        # Plot two samples
        for i in range(2):
            # Plot the RGB frames for this sample
            num_frames = 6
            fig, axs = plt.subplots(2, num_frames, figsize=(15, 3))

            # Add row titles
            axs[0, 0].set_title("Prediction", fontsize=10, loc='left', pad=10)
            axs[1, 0].set_title("Actual", fontsize=10, loc='left', pad=10)
            
            # Plot the condition frame
            condition_frame_img = condition_frame[i]
            axs[0, 0].imshow(condition_frame_img.transpose(1, 2, 0))
            axs[1, 0].imshow(condition_frame_img.transpose(1, 2, 0))
            axs[0, 0].axis('off')
            axs[1, 0].axis('off')
            
            sample_images = sample_batches[i]
            target_images = target_frames[i]

            for j in range(sample_images.shape[0]):
                axs[0,j+1].imshow(sample_images[j].transpose(1, 2, 0))
                axs[1,j+1].imshow(target_images[j].transpose(1, 2, 0))
                axs[0,j+1].axis('off')
                axs[1,j+1].axis('off')

            # Set the title of the plot
            fig.suptitle(f"Growth Rate (0 -> 2): {label[i]}")

            # Construct the file path for saving
            file_name = f"{label[i]}/plot_sample{sample_idx + i}_epoch{epoch}_process{process_idx}.png"
            file_path = os.path.join(output_dir, file_name)

            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save the figure to the specified path
            plt.savefig(file_path)
            plt.close(fig)  # Close the figure to release memory

    return file_path

# Main function
def main():
    global config

    config = TrainingConfig()

    accelerate.utils.set_seed(config.seed)

    syntheticGrowth = SyntheticGrowthDataLoader(
        config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    train_loader = syntheticGrowth.get_train_loader()
    val_loader = syntheticGrowth.get_val_loader()

    # ADM architecture
    unet = UNetCondition2D(
        sample_size=config.image_size,  # the target image resolution
        in_channels=config.image_channels,  # the number of input channels, 3 for RGB images
        out_channels=config.image_channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        encoder_hid_dim=512,
        encoder_hid_dim_type='text_proj',
        cross_attention_dim=512
    )
    controlnet = ControlUNet.from_unet(unet, conditioning_channels=config.conditioning_channels)

    optimizer = torch.optim.Adam(
        list(unet.parameters()) + list(controlnet.parameters()), 
        lr=config.learning_rate
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    controlnet_model = ControlNet(
        unet=unet,
        controlnet=controlnet,
        training_config=config,
    )

    controlnet_model.train_loop(
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=lr_scheduler,
        skip_condition=None,
        metrics=None,
        plot_function=plotter,
    )

if __name__ == '__main__':

    main()