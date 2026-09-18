import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision
import math
from torchvision.models import VGG19_Weights


class ResidualBlock(nn.Module):
    """
    Residual Block strictly following SRGAN paper
    Architecture: Conv(3x3,64) -> BN -> PReLU -> Conv(3x3,64) -> BN -> Element-wise sum
    """

    def __init__(self, n_channels=64):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            n_channels, n_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling block following SRGAN paper
    Architecture: Conv(3x3, n_channels*4) -> PixelShuffle(2x) -> PReLU
    FIXED: Removed extra Conv2d layer that was NOT in the paper
    """

    def __init__(self, in_channels, upscale_factor=2):
        super(UpsampleBlock, self).__init__()

        # Paper uses: Conv -> PixelShuffle -> PReLU (NO extra conv!)
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (upscale_factor**2),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """
    SRGAN Generator (SRResNet) strictly following the paper

    Architecture:
    1. Input Conv: Conv(9x9, 64) -> PReLU
    2. Residual Blocks: B=16 residual blocks
    3. Mid Conv: Conv(3x3, 64) -> BN (with skip connection from input)
    4. Upsampling: 2 upsample blocks (total 4x upscaling)
    5. Output: Conv(9x9, 3) -> Tanh
    """

    def __init__(self, config):
        super(Generator, self).__init__()

        large_kernel_size = config.G.large_kernel_size  # 9
        small_kernel_size = config.G.small_kernel_size  # 3
        n_channels = config.G.n_channels  # 64
        n_blocks = config.G.n_blocks  # 16
        scaling_factor = config.scaling_factor  # 4

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            3,
            n_channels,
            kernel_size=large_kernel_size,
            stride=1,
            padding=large_kernel_size // 2,
        )
        self.prelu1 = nn.PReLU()

        # B residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_channels) for _ in range(n_blocks)]
        )

        # Post-residual block convolution
        self.conv2 = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=small_kernel_size,
            stride=1,
            padding=small_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(n_channels)

        # Upsampling blocks (log2(scaling_factor) = 2 for 4x)
        num_upsample_blocks = int(math.log(scaling_factor, 2))
        self.upsample_blocks = nn.Sequential(
            *[
                UpsampleBlock(n_channels, upscale_factor=2)
                for _ in range(num_upsample_blocks)
            ]
        )

        # Output convolution
        self.conv3 = nn.Conv2d(
            n_channels,
            3,
            kernel_size=large_kernel_size,
            stride=1,
            padding=large_kernel_size // 2,
        )

    def forward(self, x):
        # First layer
        out1 = self.prelu1(self.conv1(x))

        # Residual blocks
        out = self.residual_blocks(out1)

        # Post-residual convolution + BN
        out2 = self.bn2(self.conv2(out))

        # Skip connection (element-wise sum)
        out = out1 + out2

        # Upsampling
        out = self.upsample_blocks(out)

        # Output layer with Tanh activation
        out = self.conv3(out)
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    """
    SRGAN Discriminator with MANDATORY Spectral Normalization (NO BatchNorm)

    Architecture: 8 convolutional blocks
    - Block structure: Conv(k3, nXX, sY) -> LeakyReLU(0.2)
    - Channel progression: 64, 64, 128, 128, 256, 256, 512, 512
    - Stride pattern: 1, 2, 1, 2, 1, 2, 1, 2
    - 🔥 CRITICAL FIX: Removed BatchNorm when using Spectral Norm
    - Followed by: Dense(1024) -> LeakyReLU -> Dense(1) -> Sigmoid

    ✅ FIXED: BatchNorm conflicts with Spectral Norm
    - BatchNorm + SpectralNorm = Unstable training
    - Use ONLY Spectral Norm for gradient control
    - More stable GAN training
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()

        kernel_size = config.D.kernel_size  # 3
        fc_size = config.D.fc_size  # 1024

        # Check if spectral norm is enabled (default: True)
        use_spectral_norm = config.get("use_spectral_norm", True)

        # Helper function to conditionally apply spectral norm
        def maybe_spectral_norm(layer):
            """Apply spectral normalization if enabled"""
            if use_spectral_norm:
                return spectral_norm(layer)
            return layer

        # 🔥 FIX 1: Convolutional blocks WITHOUT BatchNorm when using SpectralNorm
        # Blocks: (out_channels, stride)
        conv_blocks = [
            (64, 1),  # Block 1
            (64, 2),  # Block 2
            (128, 1),  # Block 3
            (128, 2),  # Block 4
            (256, 1),  # Block 5
            (256, 2),  # Block 6
            (512, 1),  # Block 7
            (512, 2),  # Block 8
        ]

        layers = []
        in_channels = 3

        for out_channels, stride in conv_blocks:
            # Apply spectral norm to Conv2d layers
            layers.append(
                maybe_spectral_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                    )
                )
            )
            # 🔥 CRITICAL: NO BatchNorm when using Spectral Norm
            # Only use LeakyReLU
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Dense layers with spectral norm
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # Ensure consistent spatial size
            nn.Flatten(),
            maybe_spectral_norm(nn.Linear(512 * 6 * 6, fc_size)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm(nn.Linear(fc_size, 1)),
            # Note: BCEWithLogitsLoss includes sigmoid, so no sigmoid here
        )

        # Store spectral norm status for logging
        self.use_spectral_norm = use_spectral_norm

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TruncatedVGG19(nn.Module):
    """
    Truncated VGG19 for perceptual loss calculation with AUTOMATIC ImageNet normalization

    Can extract features from different layers:
    - (i=5, j=4): High-level semantic features (φ5,4) - Original SRGAN
    - (i=2, j=2): Low-level texture features (φ2,2) - For multi-layer loss

    Usage in multi-layer VGG loss:
    - vgg_high = TruncatedVGG19(i=5, j=4)  # Semantic content
    - vgg_low = TruncatedVGG19(i=2, j=2)   # Texture details
    - loss = 0.8 * mse(vgg_high(sr), vgg_high(hr)) +
             0.2 * mse(vgg_low(sr), vgg_low(hr))

    🔥 FIX 2: Added automatic ImageNet normalization
    - Input: Images in [-1, 1] range (from Generator's tanh output)
    - Automatically converts to ImageNet normalized [0, 1] with mean/std
    """

    def __init__(self, i=5, j=4):
        super(TruncatedVGG19, self).__init__()

        vgg19 = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)

        maxpool_counter = 0
        conv_counter = 0
        layers = []

        for layer in vgg19.features.children():
            layers.append(layer)

            if isinstance(layer, nn.Conv2d):
                conv_counter += 1

            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Stop condition: after i-1 maxpools and j convs in the i-th block
            # For phi_5,4: stop after 4th conv in 5th block (before 5th maxpool)
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        self.features = nn.Sequential(*layers)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Store layer info for debugging
        self.i = i
        self.j = j

        # 🔥 FIX 2: Register ImageNet normalization parameters
        # VGG19 expects inputs normalized with these values
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input images in [-1, 1] range (from Generator's tanh output)
               or [0, 1] range (from original images)

        Returns:
            VGG19 features at specified layer (φi,j)
        """
        # 🔥 CRITICAL: Normalize input to ImageNet statistics
        # Step 1: Convert [-1, 1] or [0, 1] to [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        # Step 2: Apply ImageNet normalization
        x = (x - self.mean) / self.std

        # Step 3: Extract features
        return self.features(x)


# 🔥 NEW: Weight initialization function for discriminator reset
def weights_init(m):
    """
    Initialize network weights using He/Xavier initialization
    Used when resetting discriminator during training

    Usage:
        discriminator = Discriminator(config)
        discriminator.apply(weights_init)
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)