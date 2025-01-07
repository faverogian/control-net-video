"""
This file contains an implementation of the paper 
"Adding Conditional Control to Text-to-Image Diffusion Models" 
by Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

Reference:
Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. (2023). 
Adding Conditional Control to Text-to-Image Diffusion Models. 
Retrieved from https://arxiv.org/abs/2302.05543
"""

from comet_ml import Experiment, ExistingExperiment
import torch
import torch.nn as nn
from torch.special import expm1
import math
from accelerate import Accelerator
import os
import sys
from tqdm import tqdm
from ema_pytorch import EMA
import time

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class ControlNet(nn.Module):
    def __init__(
        self, 
        unet: nn.Module,
        controlnet: nn.Module,
        training_config: dict,
        encoder_type: str = None,  # Options: 't5', 'llava', 'nn', or None
    ):
        super().__init__()

        # Training configuration
        self.config = training_config

        # Training objective
        pred_param = self.config.pred_param
        assert pred_param in ['v', 'eps'], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        schedule = self.config.schedule
        assert schedule in ['cosine', 'shifted_cosine'], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        if schedule == 'cosine':
            self.schedule = self.logsnr_schedule_cosine
        elif schedule == 'shifted_cosine':
            self.schedule = self.logsnr_schedule_cosine_shifted
        self.noise_d = self.config.noise_d
        self.image_d = self.config.image_size

        # Classifier-free guidance scale
        self.cfg_w = self.config.cfg_w

        # Model
        assert isinstance(unet, nn.Module), "Model must be an instance of torch.nn.Module."
        self.model = unet
        self.controlnet = controlnet

        # Exponential moving average
        #self.ema = EMA(
        #    self.model,
        #    beta=0.9999,
        #    update_after_step=100,
        #    update_every=10
        #)

        # Lock the parameters of the UNet - Disable for end-to-end training
        # self.model.requires_grad_(False)

        # Optional encoder setup
        self.encoder_type = self.config.encoder_type
        if self.encoder_type == 't5':
            from transformers import T5EncoderModel, T5Tokenizer
            self.encoder = T5EncoderModel.from_pretrained("t5-base")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.null_token = self.tokenizer.pad_token_id
        elif self.encoder_type == 'clinicalt5':
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.encoder = AutoModelForSeq2SeqLM.from_pretrained("luqh/ClinicalT5-base").encoder
            self.tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base")
            self.null_token = self.tokenizer.pad_token_id
        elif self.encoder_type == 'llava':
            from transformers import AutoModel, AutoTokenizer
            self.encoder = AutoModel.from_pretrained("microsoft/llava-med-v1.5-mistral-7b").encoder
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")
            self.null_token = self.tokenizer.pad_token_id
        elif self.encoder_type == 'nn':
            classes = self.config.classes
            embedding_dim = unet.config.encoder_hid_dim
            self.encoder = nn.Embedding(classes, embedding_dim)
            self.tokenizer = None  # Not required for embeddings
            self.null_token = classes - 1  # Placeholder for null token
        else:
            self.encoder = None
            self.tokenizer = None

        # Lock the parameters of the encoder - Disable for end-to-end training
        if self.encoder in ['t5', 'llava', 'clinicalt5']:
            self.encoder.requires_grad_(False)

        num_params = sum(p.numel() for p in self.model.parameters())
        num_params += sum(p.numel() for p in self.controlnet.parameters())
        print(f"Number of parameters: {num_params}")

    def encode_text_prompt(self, text):
        """
        Encode a text prompt using the selected encoder, if available.
        """
        if self.encoder_type == 'nn':
            embeddings = self.encoder(text)
            embeddings.unsqueeze_(1)
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs.to(self.encoder.device)
            with torch.no_grad():
                embeddings = self.encoder(**inputs).last_hidden_state
        return embeddings

    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan((t_min + t * (t_max - t_min)).clone().detach()))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted
        
    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def ddpm_sampler_step(self, z_t, pred, u_pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        u_pred (torch.Tensor): The unconditional prediction from the model.
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        w = self.cfg_w
        pred = (1+w)*pred - w*u_pred # cfg_pred = (1+w)pred - w*u_pred
        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance

    @torch.no_grad()
    def sample(self, x, c, text=None):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x (torch.Tensor): An input tensor that is of the desired shape.
        c (torch.Tensor): The control tensor.

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        z_t = torch.randn(x.shape).to(x.device)
        c = c.to(x.device)

        # Get embeddings (and null embeddings) if text is provided
        if text is not None and self.encoder is not None:
            text_embeddings = self.encode_text_prompt(text)
            text_embeddings = text_embeddings.to(x.device)

            null_tokens = torch.full_like(text, self.null_token)
            null_embeddings = self.encode_text_prompt(null_tokens)
            null_embeddings = null_embeddings.to(x.device)
            null_image = torch.zeros_like(c).to(x.device)
        else:
            text_embeddings = None
            null_embeddings = None
            null_image = torch.zeros_like(c).to(x.device)

        # Create evenly spaced steps, e.g., for sampling_steps=5 -> [1.0, 0.8, 0.6, 0.4, 0.2]
        steps = torch.linspace(1.0, 0.0, self.config.sampling_steps + 1)

        for i in range(len(steps) - 1):  # Loop through steps, but stop before the last element
            
            u_t = steps[i]  # Current step
            u_s = steps[i + 1]  # Next step

            logsnr_t = self.schedule(u_t).to(x.device)
            logsnr_s = self.schedule(u_s).to(x.device)

            # Conditional sample
            down_residuals, mid_residuals = self.controlnet(z_t, logsnr_t, c, text_embeddings)
            pred = self.model(
                z_t, 
                logsnr_t,
                downblock_additional_residuals=down_residuals,
                midblock_additional_residuals=mid_residuals,
                encoder_hidden_states=text_embeddings
                )

            # Unconditional sample
            down_residuals, mid_residuals = self.controlnet(z_t, logsnr_t, null_image, null_embeddings)
            u_pred = self.model(
                z_t, 
                logsnr_t,
                downblock_additional_residuals=down_residuals,
                midblock_additional_residuals=mid_residuals,
                encoder_hidden_states=null_embeddings
                )

            mu, variance = self.ddpm_sampler_step(z_t, pred, u_pred, logsnr_t.clone().detach(), logsnr_s.clone().detach())
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        logsnr_1 = self.schedule(steps[-2]).to(x.device)
        logsnr_0 = self.schedule(steps[-1]).to(x.device)

        # Conditional sample
        down_residuals, mid_residuals = self.controlnet(z_t, logsnr_1, c, text_embeddings)
        pred = self.model(
            z_t, 
            logsnr_1,
            downblock_additional_residuals=down_residuals,
            midblock_additional_residuals=mid_residuals,
            encoder_hidden_states=text_embeddings
            )

        # Unconditional sample
        down_residuals, mid_residuals = self.controlnet(z_t, logsnr_1, null_image, null_embeddings)
        u_pred = self.model(
            z_t,
            logsnr_1,
            downblock_additional_residuals=down_residuals,
            midblock_additional_residuals=mid_residuals,
            encoder_hidden_states=null_embeddings
            )

        x_pred, _ = self.ddpm_sampler_step(z_t, pred, u_pred, logsnr_1.clone().detach(), logsnr_0.clone().detach())
        
        x_pred = self.clip(x_pred)

        # Convert x_pred to the range [0, 1]
        x_pred = (x_pred + 1) / 2

        return x_pred
    
    def loss(self, x, c, text=None):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.
        c (torch.Tensor): The control tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])
        c = c.to(x.device)

        if text is not None and self.encoder is not None:
            text_embeddings = self.encode_text_prompt(text)
            text_embeddings = text_embeddings.to(x.device)
        else:
            text_embeddings = None

        logsnr_t = self.schedule(t).to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        down_residuals, mid_residuals = self.controlnet(
            x=z_t, 
            noise_labels=logsnr_t,
            conditioning=c,
            encoder_hidden_states=text_embeddings 
            )
        pred = self.model(
            x=z_t, 
            noise_labels=logsnr_t,
            downblock_additional_residuals=down_residuals,
            midblock_additional_residuals=mid_residuals,
            encoder_hidden_states=text_embeddings,
            )

        if self.pred_param == 'v':
            eps_pred = sigma_t * z_t + alpha_t * pred
        else: 
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max = 5)
        if self.pred_param == 'v':
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr
        minsnr_weight = weight.view(-1, 1, 1, 1)

        # Get the absolute error
        error = eps_pred - eps_t

        loss = torch.mean(minsnr_weight * (error) ** 2)

        return loss

    def train_loop(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
        metrics=None,
        skip_condition=None,
        plot_function=None,
    ):
        """
        A function to train the model.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics (list): A list of metrics to evaluate.
        skip_condition (function): A function to skip samples based on a condition.
        plot_function (function): The function to use for plotting the samples.

        Returns:
        None
        """

        # Initialize CometML experiment
        experiment = None

        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=self.config.experiment_path,
        )
        
        # Prepare the model, optimizer, dataloaders, and learning rate scheduler
        controlnet, unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare( 
            self.controlnet, self.model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        if self.encoder_type == 'nn': # Learnable embeddings require separate preparation than pretrained models
            self.encoder = accelerator.prepare(self.encoder)

        # Ensure metrics are on the correct device
        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        # Check if resume training is enabled
        if self.config.resume:
            checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
            start_epoch, experiment_key = self.load_checkpoint(checkpoint_path, accelerator)
            if experiment_key is not None and self.config.use_comet and accelerator.is_main_process:
                experiment = ExistingExperiment(
                    previous_experiment=experiment_key,
                    api_key=self.config.comet_api_key,
                )
        else: # Set up fresh experiment
            if self.config.use_comet and accelerator.is_main_process:
                experiment = Experiment(
                    api_key=self.config.comet_api_key,
                    project_name=self.config.comet_project_name,
                    workspace=self.config.comet_workspace,
                )
                experiment.set_name(self.config.comet_experiment_name)
                experiment.log_asset(os.path.join(self.config.experiment_path, 'scripts/train.py'), 'train.py')
                experiment.log_asset(os.path.join(self.config.project_root, 'scripts/train.sh'), 'train.sh')
                experiment.log_asset(os.path.join(self.config.experiment_path, 'split/split.py'), 'split.py')
                experiment.log_other("GPU Model", torch.cuda.get_device_name(0))
                experiment.log_other("Python Version", sys.version)
            start_epoch = 0

        # Train!
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()

            unet.train()
            controlnet.train()

            for _, batch in enumerate(train_dataloader):

                # Skip samples based on a condition (provided in train_controlnet.py)
                if (epoch < 0.75 * self.config.num_epochs) and (skip_condition is not None):
                    skip_mask = skip_condition(batch)
                    batch = {k: v[~skip_mask] for k, v in batch.items()}

                    if batch['images'].shape[0] == 0:
                        continue

                with accelerator.accumulate((controlnet, unet)):
                    x = batch["images"]
                    c = batch["conditions"]
                    p = batch["prompt"] if "prompt" in batch.keys() else None

                    # Stochastically drop out the conditions with probability 0.2
                    p_drop = 0.15
                    if p is not None:
                        # Sometimes replace the prompt with a null token
                        mask = torch.rand_like(p.float()) < p_drop
                        p = torch.where(mask, torch.full_like(p, self.null_token), p).long()

                        # Sometimes replace the condition with a zero tensor
                        mask = torch.rand_like(p.float()) < p_drop
                        mask = mask.reshape(-1, 1, 1, 1)
                        c = torch.where(mask, torch.zeros_like(c), c)

                    loss = self.loss(x, c, p)
                    loss = loss.to(next(controlnet.parameters()).dtype)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        controlnet_params = dict(controlnet.named_parameters())
                        unet_params = dict(unet.named_parameters())
                        all_params = {**controlnet_params, **unet_params}
                        accelerator.clip_grad_norm_(all_params.values(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Update EMA model parameters
                #self.ema.update()

            epoch_elapsed = time.time() - epoch_start_time
            if accelerator.is_main_process:
                print(f"Epoch {epoch}/{self.config.num_epochs}: {epoch_elapsed:.2f} s.")

                # Log the loss to CometML
                if experiment is not None:
                    experiment.log_metric("loss", loss.item(), epoch=epoch)

            # Run an evaluation on validation set
            if epoch % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                val_evaluation_start_time = time.time()

                unet.eval()
                controlnet.eval()

                # Make directory for saving images
                training_images_path = os.path.join(self.config.experiment_path, "training_images/")

                val_samples, batches, metrics = self.evaluate(
                                                    val_dataloader, 
                                                    stop_idx=self.config.evaluation_batches,
                                                    metrics=metrics,
                                                    skip_condition=skip_condition,
                                                )
                
                # Use the provided plot_function to plot the samples
                if plot_function is not None:
                    image_path = plot_function(
                        output_dir=training_images_path,
                        batches=batches,
                        samples=val_samples, 
                        epoch=epoch, 
                        process_idx=accelerator.state.process_index
                    )

                if metrics is not None:
                    for metric in metrics:
                        metric.sync_across_processes(accelerator)
                        metric_output = metric.get_output()
                        if experiment is not None and accelerator.is_main_process:
                            print(metric_output)
                            metric_output = {f"val_{metric_name}": value for metric_name, value in metric_output.items()}
                            experiment.log_metrics(metric_output, step=epoch)
                            experiment.log_image(name=f"Sample at epoch {epoch}", image_data=image_path)
                        metric.reset()
                
                val_evaluation_elapsed = time.time() - val_evaluation_start_time
                if accelerator.is_main_process:
                    self.save_checkpoint(accelerator, epoch, metrics, experiment)

                    print(f"Val evaluation time: {val_evaluation_elapsed:.2f} s.")

                unet.train()
                controlnet.train()
                    
    @torch.no_grad()
    def evaluate(
        self, 
        val_dataloader, 
        stop_idx=None,
        metrics=None,
        skip_condition=None,
    ):
        """
        A function to evaluate the model.

        Args:
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        stop_idx (int): The index to stop at.
        metrics (list): A list of metrics to evaluate.
        skip_condition (function): A function to skip conditions.
        """
        
        val_samples = []
        batches = []

        # Make a progress bar
        progress_bar = tqdm(val_dataloader, desc="Evaluating")
        for idx, batch in enumerate(val_dataloader):
            progress_bar.update(1)

            batch = {k: v for k, v in batch.items()}

            if skip_condition is not None:
                skip_mask = skip_condition(batch)
                batch = {k: v[~skip_mask] for k, v in batch.items()}
                
                if batch['images'].shape[0] == 0:
                    continue

            sample_shape = batch["images"].shape
            samples = torch.zeros((self.config.inference_acc_steps,) + sample_shape)
            for i in range(self.config.inference_acc_steps):
                x = batch["images"]
                c = batch["conditions"]
                p = batch["prompt"] if "prompt" in batch.keys() else None
                sample = self.sample(x, c, p)
                samples[i] = sample

            # Average the samples
            sample = torch.mean(samples, dim=0)
            
            # Update the metrics
            if metrics is not None:
                for metric in metrics:
                    metric.update((sample, batch))

            val_samples.append(sample)
            batches.append(batch)

            if stop_idx is not None and idx == stop_idx:
                break

        progress_bar.close()

        return val_samples, batches, metrics

    @torch.no_grad()
    def inference(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
        metrics=None,
        skip_condition=None,
        plot_function=None,
    ):
        """
        A function to perform inference.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics (list): A list of metrics to evaluate.
        skip_condition (function): A function to skip conditions.
        plot_function (function): The function to use for plotting the samples.
        """

        # Make directory for saving images
        inference_image_path = os.path.join(self.config.experiment_path, "inference_images/")
        os.makedirs(inference_image_path, exist_ok=True)

        # Make accelerator wrapper
        accelerator = Accelerator()

        controlnet, unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare( 
            self.controlnet, self.model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        if self.encoder_type == 'nn': # Learnable embeddings require separate preparation than pretrained models
            self.encoder = accelerator.prepare(self.encoder)
        
        # Load most recent checkpoint
        checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
        self.load_checkpoint(checkpoint_path, accelerator)

        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        val_samples, batches, metrics = self.evaluate(
                                            val_dataloader, 
                                            metrics=metrics,
                                            skip_condition=skip_condition,
                                            stop_idx=self.config.evaluation_batches,
                                        )
        
        metric_output = []
        if metrics is not None:
            for metric in metrics:
                metric.sync_across_processes(accelerator)
                metric_output.append(metric.get_output())

        # Use the provided plot_function to plot the samples
        if plot_function is not None:
            image_path = plot_function(
                output_dir=inference_image_path,
                batches=batches,
                samples=val_samples, 
                epoch=0, 
                process_idx=accelerator.state.process_index
            )

        return (metric_output, val_samples, batches) if metrics is not None else (val_samples, batches)
    
    def save_checkpoint(self, accelerator: Accelerator, epoch, metrics, experiment):
        """
        Saves the model checkpoint.

        Args:
        accelerator (accelerate.Accelerator): The Accelerator object.
        epoch (int): The current epoch.
        metrics (list): A list of metrics.
        experiment (comet_ml.Experiment): The CometML experiment object.
        """
        checkpoint_dir = os.path.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

        # Save accelerator state
        accelerator.save_state(output_dir=checkpoint_dir)

        # Save experiment state
        experiment_state = {
            'epoch': epoch + 1,
            'experiment_key': experiment.get_key() if experiment is not None else None
        }

        # Save new checkpoint
        latest_exp_state_path = os.path.join(checkpoint_dir, "experiment_state.pth")
        torch.save(experiment_state, latest_exp_state_path)

        print(f"Checkpoint saved to {latest_exp_state_path}")
    
    def load_checkpoint(self, checkpoint_path, accelerator: Accelerator):
        """
        Loads the model checkpoint.

        Args:
        checkpoint_path (str): The path to the checkpoint file.
        optimizer (torch.optim): The optimizer.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        accelerator (accelerate.Accelerator): The Accelerator object.

        Returns:
        tuple: The epoch and experiment.
        """
        # Restore the accelerator state
        accelerator.load_state(input_dir=checkpoint_path)

        # Get the experiment state path
        experiment_state_path = os.path.join(checkpoint_path, "experiment_state.pth")

        # Load the checkpoint file on CPU
        checkpoint = torch.load(experiment_state_path, map_location='cpu', weights_only=False)

        # Restore the epoch to resume training from
        epoch = checkpoint['epoch']

        # Optionally, resume the experiment from Comet (if using Comet for tracking)
        experiment_key = checkpoint['experiment_key']

        print(f"Checkpoint loaded. Resuming from epoch {epoch}.")

        return epoch, experiment_key