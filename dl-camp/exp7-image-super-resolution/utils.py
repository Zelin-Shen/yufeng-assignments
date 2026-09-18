from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as F
import torch
import math


def convert_image(img, source, target, device):
    """
    Convert an image from a source format to a target format.

    Args:
        img: Input image (PIL Image or Tensor)
        source: Source format ('pil', '[0, 1]', '[-1, 1]')
        target: Target format ('pil', '[0, 255]', '[0, 1]', '[-1, 1]',
                               'imagenet-norm', 'y-channel')
        device: Target device (torch.device)

    Returns:
        Converted image (PIL Image or Tensor)
    """
    assert source in {"pil", "[0, 1]", "[-1, 1]"}, (
        f"Cannot convert from source format {source}!"
    )
    assert target in {
        "pil",
        "[0, 255]",
        "[0, 1]",
        "[-1, 1]",
        "imagenet-norm",
        "y-channel",
    }, f"Cannot convert to target format {target}!"

    # Some constants (will be moved to correct device as needed)
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966])
    imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

    # Convert from source to [0, 1] range
    if source == "pil":
        # PIL Image -> Tensor [0, 1] range
        img = F.to_tensor(img)
        # ✅ CRITICAL FIX: Move to target device immediately
        img = img.to(device)

    elif source == "[-1, 1]":
        # Tensor [-1, 1] -> [0, 1]
        img = (img + 1.0) / 2.0

    elif source == "[0, 1]":
        # Already in [0, 1] range
        pass

    # Special case: if target is PIL, move to CPU first
    if target == "pil":
        if img.device.type != "cpu":
            img = img.cpu()
        img = F.to_pil_image(img)
        return img

    # Convert from [0, 1] to target format
    if target == "[0, 255]":
        img = 255.0 * img

    elif target == "[-1, 1]":
        img = 2.0 * img - 1.0

    elif target == "[0, 1]":
        pass

    elif target == "imagenet-norm":
        # Normalize with ImageNet statistics
        if img.ndimension() == 3:
            # Single image: (C, H, W)
            imagenet_mean = imagenet_mean.to(img.device)
            imagenet_std = imagenet_std.to(img.device)
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            # Batch: (B, C, H, W)
            imagenet_mean_cuda = imagenet_mean.unsqueeze(0).to(img.device)
            imagenet_std_cuda = imagenet_std.unsqueeze(0).to(img.device)
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == "y-channel":
        # Convert to Y channel (luminance) for PSNR/SSIM calculation
        rgb_weights = rgb_weights.to(img.device)
        img = (
            torch.matmul(255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights)
            / 255.0
            + 16.0
        )

    return img


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    Args:
        optimizer: Optimizer whose learning rate must be shrunk
        shrink_factor: Factor in interval (0, 1) to multiply learning rate with
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print(f"The new learning rate is {optimizer.param_groups[0]['lr']:.6f}\n")


# ✅ NEW: Gradient monitoring function
def get_gradient_norm(model):
    """
    Calculate average gradient norm for monitoring training stability

    Args:
        model: PyTorch model (Generator or Discriminator)

    Returns:
        float: Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


# ✅ NEW: Training metrics logger
def log_training_metrics(epoch, metrics, log_file="logs/training_log.txt"):
    """
    Log training metrics to file

    Args:
        epoch: Current epoch number
        metrics: Dictionary of metrics to log
        log_file: Path to log file
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        log_line = f"Epoch {epoch}: "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_line += f"{key}={value:.6f}, "
            else:
                log_line += f"{key}={value}, "
        f.write(log_line.rstrip(", ") + "\n")


# ✅ NEW: Calculate PSNR
def calculate_psnr(img1, img2, max_val=255.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        max_val: Maximum possible pixel value (default: 255.0)

    Returns:
        float: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


# ✅ NEW: Safe gradient clipping
def clip_gradients(model, max_norm):
    """
    Safely clip gradients to prevent explosion

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm threshold

    Returns:
        float: Total norm before clipping
    """
    if max_norm <= 0:
        return 0.0

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm


# ✅ NEW: Check for NaN in tensors
def check_nan(tensor, name="tensor"):
    """
    Check if tensor contains NaN values and raise error if found

    Args:
        tensor: PyTorch tensor to check
        name: Name of tensor for error message

    Raises:
        ValueError: If NaN values are detected
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}! Training stopped.")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}! Training stopped.")


# ✅ NEW: Learning rate warmup scheduler
def get_warmup_lr(base_lr, current_epoch, warmup_epochs):
    """
    Calculate learning rate with warmup

    Args:
        base_lr: Target learning rate after warmup
        current_epoch: Current epoch (0-indexed)
        warmup_epochs: Number of warmup epochs

    Returns:
        float: Adjusted learning rate
    """
    if current_epoch >= warmup_epochs:
        return base_lr
    return base_lr * (current_epoch + 1) / warmup_epochs


# ✅ NEW: Model parameter counting
def count_parameters(model):
    """
    Count total and trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ✅ NEW: Print model summary
def print_model_summary(model, model_name="Model"):
    """
    Print detailed model summary

    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    total, trainable = count_parameters(model)
    print(f"\n{'=' * 60}")
    print(f"{model_name} Summary")
    print(f"{'=' * 60}")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size (MB):      {total * 4 / 1024 / 1024:.2f}")
    print(f"{'=' * 60}\n")


# ✅ NEW: Save sample images during training
def save_sample_images(lr_imgs, sr_imgs, hr_imgs, epoch, save_dir="samples"):
    """
    Save sample images for visual inspection during training

    Args:
        lr_imgs: Low-resolution images (B, C, H, W) tensor
        sr_imgs: Super-resolved images (B, C, H, W) tensor
        hr_imgs: High-resolution images (B, C, H, W) tensor
        epoch: Current epoch number
        save_dir: Directory to save images
    """
    os.makedirs(save_dir, exist_ok=True)

    # Take first image from batch
    lr_img = lr_imgs[0].cpu()
    sr_img = sr_imgs[0].cpu()
    hr_img = hr_imgs[0].cpu()

    # Convert to PIL images
    lr_pil = F.to_pil_image(lr_img)
    sr_pil = F.to_pil_image(torch.clamp(sr_img, 0, 1))
    hr_pil = F.to_pil_image(hr_img)

    # Resize LR image for comparison
    lr_pil_resized = lr_pil.resize(hr_pil.size, Image.BICUBIC)

    # Create comparison image
    from PIL import Image as PILImage

    width, height = hr_pil.size
    comparison = PILImage.new("RGB", (width * 3, height))
    comparison.paste(lr_pil_resized, (0, 0))
    comparison.paste(sr_pil, (width, 0))
    comparison.paste(hr_pil, (width * 2, 0))

    # Save
    comparison.save(os.path.join(save_dir, f"epoch_{epoch:04d}.png"))


# ✅ NEW: Calculate Gram matrix for style loss
def gram_matrix(tensor):
    """
    Calculate Gram matrix for style loss

    Args:
        tensor: Feature maps (B, C, H, W)

    Returns:
        Gram matrix (B, C, C)
    """
    B, C, H, W = tensor.size()
    features = tensor.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (C * H * W)


# ✅ NEW: Early stopping helper
class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """

    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss

        Returns:
            bool: True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


# ✅ NEW: Exponential moving average for smoother metrics
class ExponentialMovingAverage:
    """
    Exponential moving average for smoothing metrics
    """

    def __init__(self, decay=0.99):
        """
        Args:
            decay: Smoothing factor (higher = smoother)
        """
        self.decay = decay
        self.value = None

    def update(self, new_value):
        """
        Update EMA with new value

        Args:
            new_value: New metric value

        Returns:
            float: Smoothed value
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value