import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from models import Generator, Discriminator, TruncatedVGG19
from utils import *
import copy
import numpy as np


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights on the SAME device as model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        """Update shadow weights with exponential moving average"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Ensure shadow is on the same device
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)

                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone().detach()

    def apply_shadow(self):
        """Apply shadow weights to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original weights (after inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def warmup_lr(optimizer, current_step, warmup_steps, base_lr):
    """Cosine learning rate warmup for smooth training initialization."""
    if current_step < warmup_steps:
        progress = current_step / warmup_steps
        lr = base_lr * (0.5 * (1 + np.cos(np.pi * (1 - progress))))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    return None


def pretrain_generator(
    train_loader, generator, optimizer_g, epoch, device, config, ema=None
):
    """
    Stage 1: Pre-train Generator with HYBRID MSE + Perceptual Loss
    Combines MSE (80%) and VGG (20%) loss, with warmup, gradient accumulation and EMA.
    """
    generator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_mse = AverageMeter()
    losses_vgg = AverageMeter()
    losses_total = AverageMeter()

    mse_criterion = nn.MSELoss().to(device)

    use_vgg_pretrain = getattr(config, "use_vgg_pretrain", True)
    if use_vgg_pretrain:
        truncated_vgg19 = TruncatedVGG19(i=5, j=4).to(device)
        truncated_vgg19.eval()
        vgg_weight = getattr(config, "vgg_weight_pretrain", 0.2)

    scaler = GradScaler() if getattr(config, "use_amp", False) else None

    # Learning rate warmup settings
    warmup_epochs = getattr(config, "warmup_epochs", 5)
    warmup_steps = warmup_epochs * len(train_loader)
    base_lr = optimizer_g.param_groups[0]["lr"]
    current_step = epoch * len(train_loader)

    # Gradient accumulation for effective larger batch size
    accumulation_steps = getattr(config, "gradient_accumulation", 1)

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Apply learning rate warmup
        if epoch < warmup_epochs:
            current_lr = warmup_lr(optimizer_g, current_step + i, warmup_steps, base_lr)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        batch_size = lr_imgs.size(0)

        # Data augmentation: Random horizontal flip
        if getattr(config, "use_augmentation", True) and torch.rand(1) > 0.5:
            lr_imgs = torch.flip(lr_imgs, dims=[3])
            hr_imgs = torch.flip(hr_imgs, dims=[3])

        # Image normalization
        lr_imgs = convert_image(
            lr_imgs, source="[0, 1]", target="imagenet-norm", device=device
        )
        hr_imgs_tanh = convert_image(
            hr_imgs, source="[0, 1]", target="[-1, 1]", device=device
        )

        # Mixed precision training
        if scaler:
            with autocast():
                sr_imgs = generator(lr_imgs)

                # Primary MSE loss
                mse_loss = mse_criterion(sr_imgs, hr_imgs_tanh)

                # Auxiliary VGG perceptual loss
                if use_vgg_pretrain:
                    hr_imgs_norm = convert_image(
                        hr_imgs, source="[0, 1]", target="imagenet-norm", device=device
                    )
                    sr_imgs_norm = convert_image(
                        sr_imgs, source="[-1, 1]", target="imagenet-norm", device=device
                    )

                    with torch.no_grad():
                        hr_features = truncated_vgg19(hr_imgs_norm)
                    sr_features = truncated_vgg19(sr_imgs_norm)

                    vgg_loss = mse_criterion(sr_features, hr_features)
                    total_loss = (1 - vgg_weight) * mse_loss + vgg_weight * vgg_loss
                else:
                    vgg_loss = torch.tensor(0.0)
                    total_loss = mse_loss

                # Scale loss for gradient accumulation
                total_loss = total_loss / accumulation_steps

            scaler.scale(total_loss).backward()

            # Update parameters only at accumulation boundary
            if (i + 1) % accumulation_steps == 0:
                grad_clip = getattr(config, "grad_clip", None)
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer_g)
                    nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                scaler.step(optimizer_g)
                scaler.update()
                optimizer_g.zero_grad()

                # Update EMA weights
                if ema is not None:
                    ema.update()
        else:
            sr_imgs = generator(lr_imgs)
            mse_loss = mse_criterion(sr_imgs, hr_imgs_tanh)

            if use_vgg_pretrain:
                hr_imgs_norm = convert_image(
                    hr_imgs, source="[0, 1]", target="imagenet-norm", device=device
                )
                sr_imgs_norm = convert_image(
                    sr_imgs, source="[-1, 1]", target="imagenet-norm", device=device
                )

                with torch.no_grad():
                    hr_features = truncated_vgg19(hr_imgs_norm)
                sr_features = truncated_vgg19(sr_imgs_norm)

                vgg_loss = mse_criterion(sr_features, hr_features)
                total_loss = (1 - vgg_weight) * mse_loss + vgg_weight * vgg_loss
            else:
                vgg_loss = torch.tensor(0.0)
                total_loss = mse_loss

            total_loss = total_loss / accumulation_steps
            total_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                grad_clip = getattr(config, "grad_clip", None)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                optimizer_g.step()
                optimizer_g.zero_grad()

                if ema is not None:
                    ema.update()

        losses_mse.update(mse_loss.item(), batch_size)
        if use_vgg_pretrain:
            losses_vgg.update(vgg_loss.item(), batch_size)
        losses_total.update(total_loss.item() * accumulation_steps, batch_size)
        batch_time.update(time.time() - start)
        start = time.time()

        # Print training progress
        if i % config.print_freq == 0:
            current_lr_str = (
                f"{current_lr:.6f}"
                if epoch < warmup_epochs
                else f"{optimizer_g.param_groups[0]['lr']:.6f}"
            )

            if use_vgg_pretrain:
                print(
                    "Pretrain Epoch: [{0}][{1}/{2}]|"
                    "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)|"
                    "Data {data_time.val:.3f}s ({data_time.avg:.3f}s)|"
                    "LR {lr}|"
                    "MSE {loss_mse.val:.4f} ({loss_mse.avg:.4f})|"
                    "VGG {loss_vgg.val:.4f} ({loss_vgg.avg:.4f})|"
                    "Total {loss_total.val:.4f} ({loss_total.avg:.4f})".format(
                        epoch + 1,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        lr=current_lr_str,
                        loss_mse=losses_mse,
                        loss_vgg=losses_vgg,
                        loss_total=losses_total,
                    )
                )
            else:
                print(
                    "Pretrain Epoch: [{0}][{1}/{2}]|"
                    "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)|"
                    "LR {lr}|"
                    "MSE Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                        epoch + 1,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        lr=current_lr_str,
                        loss=losses_mse,
                    )
                )

    # Memory cleanup
    del lr_imgs, hr_imgs, sr_imgs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return losses_total.avg


def train_gan(
    train_loader,
    generator,
    discriminator,
    truncated_vgg19,
    content_loss_criterion,
    adversarial_loss_criterion,
    optimizer_g,
    optimizer_d,
    epoch,
    device,
    config,
    ema=None,
):
    """
    Stage 2: GAN Training with FIXED adaptive discriminator control

    🔥 KEY FIXES:
    1. ✅ Discriminator never completely stops updating (forced periodic updates)
    2. ✅ Reduced accuracy threshold from 0.95 to 0.92
    3. ✅ Increased beta dynamically based on training phase
    4. ✅ Added forced update every 5 batches to prevent saturation
    """
    generator.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_c = AverageMeter()
    losses_a_g = AverageMeter()
    losses_d = AverageMeter()
    losses_g = AverageMeter()
    d_real_acc = AverageMeter()  # D accuracy on real images
    d_fake_acc = AverageMeter()  # D accuracy on fake images

    scaler = GradScaler() if getattr(config, "use_amp", False) else None

    # 🔥 FIX 1: Progressive beta with higher base values
    base_beta = getattr(config, "beta", 1e-3)
    if epoch < 10:
        beta = base_beta * 0.5  # First 10 epochs: 50% adversarial (was 10%)
    elif epoch < 30:
        beta = base_beta * 1.0  # Epoch 10-30: 100% adversarial (was 50%)
    else:
        beta = base_beta * 2.0  # After epoch 30: 200% adversarial (NEW!)

    # Label smoothing + noisy labels for D robustness
    use_label_smoothing = getattr(config, "use_label_smoothing", True)
    real_label_value = getattr(config, "real_label_smoothing", 0.9)
    label_noise_prob = getattr(config, "label_noise_prob", 0.05)

    # R1 gradient penalty settings
    use_gradient_penalty = getattr(config, "use_gradient_penalty", True)
    lambda_gp = getattr(config, "lambda_gp", 10.0)

    # 🔥 FIX 2: Adjusted thresholds for more balanced training
    d_loss_threshold_low = 0.05  # was 0.1 - now more conservative
    d_loss_threshold_high = 1.5  # was 2.0 - now triggers earlier
    target_d_accuracy = 0.75  # was 0.8 - now targets lower accuracy

    # Multi-layer VGG loss setup
    use_multilayer_vgg = getattr(config, "use_multilayer_vgg", True)
    if use_multilayer_vgg:
        vgg_layer2 = getattr(config, "truncated_vgg19_low", None)

        if vgg_layer2 is None:
            print(
                "⚠️ Warning: truncated_vgg19_low not found in config, creating new instance"
            )
            vgg_layer2 = TruncatedVGG19(i=2, j=2).to(device)
            vgg_layer2.eval()
            config.truncated_vgg19_low = vgg_layer2

        vgg_weight_low = getattr(config, "vgg_weight_low", 0.2)
        vgg_weight_high = getattr(config, "vgg_weight_high", 0.8)

    # Style loss (Gram matrix) for texture enhancement
    use_style_loss = getattr(config, "use_style_loss", False)
    style_weight = getattr(config, "style_weight", 0.1)

    # Gradient accumulation
    accumulation_steps = getattr(config, "gradient_accumulation", 1)

    # Training counters
    d_update_counter = 0
    d_skip_counter = 0
    g_better_counter = 0  # Times G successfully fools D

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        batch_size = lr_imgs.size(0)

        # Data augmentation: random horizontal flip
        if getattr(config, "use_augmentation", True) and torch.rand(1) > 0.5:
            lr_imgs = torch.flip(lr_imgs, dims=[3])
            hr_imgs = torch.flip(hr_imgs, dims=[3])

        # Image normalization
        lr_imgs = convert_image(
            lr_imgs, source="[0, 1]", target="imagenet-norm", device=device
        )
        hr_imgs = convert_image(
            hr_imgs, source="[0, 1]", target="imagenet-norm", device=device
        )

        # ========================================
        # 🔥 FIXED ADAPTIVE DISCRIMINATOR TRAINING
        # ========================================

        # Generate SR images for evaluation
        with torch.no_grad():
            sr_imgs = generator(lr_imgs)
            sr_imgs_norm = convert_image(
                sr_imgs, source="[-1, 1]", target="imagenet-norm", device=device
            )

            # Evaluate current D state
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs_norm)

            # Calculate accuracy metrics
            d_real_pred = (torch.sigmoid(hr_discriminated) > 0.5).float().mean().item()
            d_fake_pred = (torch.sigmoid(sr_discriminated) < 0.5).float().mean().item()
            current_d_accuracy = (d_real_pred + d_fake_pred) / 2

            # Prepare labels
            if use_label_smoothing:
                real_labels = torch.ones_like(hr_discriminated) * real_label_value
            else:
                real_labels = torch.ones_like(hr_discriminated)

            # Apply noisy labels occasionally
            if torch.rand(1) < label_noise_prob:
                real_labels = 1 - real_labels

            fake_labels = torch.zeros_like(sr_discriminated)
            if torch.rand(1) < label_noise_prob:
                fake_labels = 1 - fake_labels

            d_loss_real = adversarial_loss_criterion(hr_discriminated, real_labels)
            d_loss_fake = adversarial_loss_criterion(sr_discriminated, fake_labels)
            d_loss_estimate = (d_loss_real + d_loss_fake).item()

        # 🔥 FIX 3: New discriminator update strategy (CRITICAL!)
        should_update_d = False
        d_update_reason = ""

        # Strategy 1: FORCED periodic updates (prevents complete saturation)
        if i % 5 == 0:
            should_update_d = True
            d_update_reason = "FORCED_PERIODIC"

        # Strategy 2: Boost weak discriminator
        elif current_d_accuracy < 0.65:  # was 0.6
            should_update_d = True
            d_update_reason = "D_WEAK_BOOST"

        # Strategy 3: Slow down strong discriminator (BUT NEVER STOP!)
        elif current_d_accuracy > 0.92:  # was 0.95 - lowered threshold
            # Even when strong, update every 15 batches
            if i % 15 == 0:
                should_update_d = True
                d_update_reason = "D_STRONG_SLOW"
            else:
                should_update_d = False
                d_skip_counter += 1
                d_update_reason = "D_STRONG_SKIP"

        # Strategy 4: Loss-based decisions in normal range
        elif d_loss_estimate > d_loss_threshold_high:
            should_update_d = True
            d_update_reason = "D_HIGH_LOSS"

        elif d_loss_estimate < d_loss_threshold_low:
            # Low loss but not too accurate: scheduled updates
            should_update_d = i % 8 == 0  # was 10 - more frequent
            d_update_reason = (
                "D_LOW_LOSS_SCHED" if should_update_d else "D_LOW_LOSS_SKIP"
            )
            if not should_update_d:
                d_skip_counter += 1

        else:
            # Normal range: update every 3 batches
            should_update_d = i % 3 == 0
            d_update_reason = "NORMAL" if should_update_d else "NORMAL_SKIP"
            if not should_update_d:
                d_skip_counter += 1

        # Execute discriminator update
        if should_update_d:
            if scaler:
                with autocast():
                    hr_discriminated = discriminator(hr_imgs)
                    sr_discriminated = discriminator(sr_imgs_norm.detach())

                    # Regenerate labels (avoid using cached noisy labels)
                    if use_label_smoothing:
                        real_labels = (
                            torch.ones_like(hr_discriminated) * real_label_value
                        )
                    else:
                        real_labels = torch.ones_like(hr_discriminated)
                    fake_labels = torch.zeros_like(sr_discriminated)

                    d_loss_real = adversarial_loss_criterion(
                        hr_discriminated, real_labels
                    )
                    d_loss_fake = adversarial_loss_criterion(
                        sr_discriminated, fake_labels
                    )
                    d_loss = d_loss_real + d_loss_fake

                    # R1 Gradient Penalty (computed every 16 steps)
                    if use_gradient_penalty and (i % 16 == 0):
                        hr_imgs.requires_grad_(True)
                        hr_discriminated_gp = discriminator(hr_imgs)

                        gradients = torch.autograd.grad(
                            outputs=hr_discriminated_gp.sum(),
                            inputs=hr_imgs,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )[0]

                        gradient_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
                        d_loss = d_loss + lambda_gp * gradient_penalty
                        hr_imgs.requires_grad_(False)

                    d_loss = d_loss / accumulation_steps

                scaler.scale(d_loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    grad_clip = getattr(config, "grad_clip", None)
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer_d)
                        nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)

                    scaler.step(optimizer_d)
                    scaler.update()
                    optimizer_d.zero_grad()
            else:
                hr_discriminated = discriminator(hr_imgs)
                sr_discriminated = discriminator(sr_imgs_norm.detach())

                if use_label_smoothing:
                    real_labels = torch.ones_like(hr_discriminated) * real_label_value
                else:
                    real_labels = torch.ones_like(hr_discriminated)
                fake_labels = torch.zeros_like(sr_discriminated)

                d_loss_real = adversarial_loss_criterion(hr_discriminated, real_labels)
                d_loss_fake = adversarial_loss_criterion(sr_discriminated, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                if use_gradient_penalty and (i % 16 == 0):
                    hr_imgs.requires_grad_(True)
                    hr_discriminated_gp = discriminator(hr_imgs)

                    gradients = torch.autograd.grad(
                        outputs=hr_discriminated_gp.sum(),
                        inputs=hr_imgs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]

                    gradient_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
                    d_loss = d_loss + lambda_gp * gradient_penalty
                    hr_imgs.requires_grad_(False)

                d_loss = d_loss / accumulation_steps
                d_loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    grad_clip = getattr(config, "grad_clip", None)
                    if grad_clip and grad_clip > 0:
                        nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)

                    optimizer_d.step()
                    optimizer_d.zero_grad()

            d_update_counter += 1
        else:
            d_loss = torch.tensor(d_loss_estimate)

        losses_d.update(d_loss.item() * accumulation_steps, batch_size)
        d_real_acc.update(d_real_pred, batch_size)
        d_fake_acc.update(d_fake_pred, batch_size)

        # ========================================
        # TRAIN GENERATOR (ALWAYS)
        # ========================================

        if scaler:
            with autocast():
                sr_imgs = generator(lr_imgs)
                sr_imgs_norm = convert_image(
                    sr_imgs, source="[-1, 1]", target="imagenet-norm", device=device
                )

                # Multi-layer VGG perceptual loss
                if use_multilayer_vgg and vgg_layer2 is not None:
                    # High-level features (phi_5,4)
                    sr_features_high = truncated_vgg19(sr_imgs_norm)
                    hr_features_high = truncated_vgg19(hr_imgs).detach()
                    content_loss_high = content_loss_criterion(
                        sr_features_high, hr_features_high
                    )

                    # Low-level features (phi_2,2)
                    sr_features_low = vgg_layer2(sr_imgs_norm)
                    hr_features_low = vgg_layer2(hr_imgs).detach()
                    content_loss_low = content_loss_criterion(
                        sr_features_low, hr_features_low
                    )

                    # Combined perceptual loss
                    content_loss = (
                        vgg_weight_high * content_loss_high
                        + vgg_weight_low * content_loss_low
                    )
                else:
                    sr_features = truncated_vgg19(sr_imgs_norm)
                    hr_features = truncated_vgg19(hr_imgs).detach()
                    content_loss = content_loss_criterion(sr_features, hr_features)

                # Style loss (Gram matrix) for texture
                if use_style_loss:

                    def gram_matrix(features):
                        b, c, h, w = features.size()
                        features = features.view(b, c, h * w)
                        gram = torch.bmm(features, features.transpose(1, 2))
                        return gram / (c * h * w)

                    sr_gram = gram_matrix(
                        sr_features_high if use_multilayer_vgg else sr_features
                    )
                    hr_gram = gram_matrix(
                        hr_features_high if use_multilayer_vgg else hr_features
                    )
                    style_loss = content_loss_criterion(sr_gram, hr_gram)
                    content_loss = content_loss + style_weight * style_loss

                # Adversarial loss
                sr_discriminated = discriminator(sr_imgs_norm)
                adversarial_loss = adversarial_loss_criterion(
                    sr_discriminated, torch.ones_like(sr_discriminated)
                )

                # 🔥 FIX 4: Adaptive loss clipping with higher limits
                adv_loss_max = 15.0 if epoch < 50 else 8.0  # was 10.0/5.0
                adversarial_loss = torch.clamp(adversarial_loss, max=adv_loss_max)

                # 🔥 FIX 5: More aggressive dynamic beta adjustment
                current_ratio = content_loss.item() / (adversarial_loss.item() + 1e-8)
                if current_ratio < 0.3:  # was 0.5 - more aggressive
                    dynamic_beta = beta * 3.0  # was 2.0
                elif current_ratio > 5.0:
                    dynamic_beta = beta * 0.5
                elif current_ratio > 2.0:  # NEW: gentle reduction for moderate ratios
                    dynamic_beta = beta * 0.8
                else:
                    dynamic_beta = beta

                # Total generator loss
                perceptual_loss = content_loss + dynamic_beta * adversarial_loss
                perceptual_loss = perceptual_loss / accumulation_steps

            scaler.scale(perceptual_loss).backward()

            if (i + 1) % accumulation_steps == 0:
                grad_clip = getattr(config, "grad_clip", None)
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer_g)
                    nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                scaler.step(optimizer_g)
                scaler.update()
                optimizer_g.zero_grad()

                if ema is not None:
                    ema.update()
        else:
            sr_imgs = generator(lr_imgs)
            sr_imgs_norm = convert_image(
                sr_imgs, source="[-1, 1]", target="imagenet-norm", device=device
            )

            # Multi-layer VGG perceptual loss (non-AMP)
            if use_multilayer_vgg and vgg_layer2 is not None:
                sr_features_high = truncated_vgg19(sr_imgs_norm)
                hr_features_high = truncated_vgg19(hr_imgs).detach()
                content_loss_high = content_loss_criterion(
                    sr_features_high, hr_features_high
                )

                sr_features_low = vgg_layer2(sr_imgs_norm)
                hr_features_low = vgg_layer2(hr_imgs).detach()
                content_loss_low = content_loss_criterion(
                    sr_features_low, hr_features_low
                )

                content_loss = (
                    vgg_weight_high * content_loss_high
                    + vgg_weight_low * content_loss_low
                )
            else:
                sr_features = truncated_vgg19(sr_imgs_norm)
                hr_features = truncated_vgg19(hr_imgs).detach()
                content_loss = content_loss_criterion(sr_features, hr_features)

            # Style loss (Gram matrix) for texture (non-AMP)
            if use_style_loss:

                def gram_matrix(features):
                    b, c, h, w = features.size()
                    features = features.view(b, c, h * w)
                    gram = torch.bmm(features, features.transpose(1, 2))
                    return gram / (c * h * w)

                sr_gram = gram_matrix(
                    sr_features_high if use_multilayer_vgg else sr_features
                )
                hr_gram = gram_matrix(
                    hr_features_high if use_multilayer_vgg else hr_features
                )
                style_loss = content_loss_criterion(sr_gram, hr_gram)
                content_loss = content_loss + style_weight * style_loss

            # Adversarial loss (non-AMP)
            sr_discriminated = discriminator(sr_imgs_norm)
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated)
            )

            # Adaptive loss clipping
            adv_loss_max = 15.0 if epoch < 50 else 8.0
            adversarial_loss = torch.clamp(adversarial_loss, max=adv_loss_max)

            # Dynamic beta adjustment
            current_ratio = content_loss.item() / (adversarial_loss.item() + 1e-8)
            if current_ratio < 0.3:
                dynamic_beta = beta * 3.0
            elif current_ratio > 5.0:
                dynamic_beta = beta * 0.5
            elif current_ratio > 2.0:
                dynamic_beta = beta * 0.8
            else:
                dynamic_beta = beta

            # Total generator loss
            perceptual_loss = content_loss + dynamic_beta * adversarial_loss
            perceptual_loss = perceptual_loss / accumulation_steps
            perceptual_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                grad_clip = getattr(config, "grad_clip", None)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)

                optimizer_g.step()
                optimizer_g.zero_grad()

                if ema is not None:
                    ema.update()

        # Track G success rate at fooling D
        with torch.no_grad():
            g_success_rate = (
                (torch.sigmoid(sr_discriminated) > 0.5).float().mean().item()
            )
            if g_success_rate > 0.5:
                g_better_counter += 1

        losses_c.update(content_loss.item(), batch_size)
        losses_a_g.update(adversarial_loss.item(), batch_size)
        losses_g.update(perceptual_loss.item() * accumulation_steps, batch_size)

        batch_time.update(time.time() - start)
        start = time.time()

        # Print training progress
        if i % config.print_freq == 0:
            print(
                "GAN Epoch: [{0}][{1}/{2}]|"
                "Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)|"
                "D_Update: {d_update} ({d_reason})|"
                "D_Loss: {loss_d.val:.4f} ({loss_d.avg:.4f})|"
                "Content: {loss_c.val:.4f} ({loss_c.avg:.4f})|"
                "Adv_G: {loss_a.val:.4f} ({loss_a.avg:.4f})|"
                "G_Total: {loss_g.val:.4f} ({loss_g.avg:.4f})|"
                "D_Acc: Real={d_real:.2%} Fake={d_fake:.2%}|"
                "Beta: {beta:.5f}".format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    d_update="YES" if should_update_d else "NO",
                    d_reason=d_update_reason,
                    loss_d=losses_d,
                    loss_c=losses_c,
                    loss_a=losses_a_g,
                    loss_g=losses_g,
                    d_real=d_real_acc.avg,
                    d_fake=d_fake_acc.avg,
                    beta=dynamic_beta,
                )
            )

    # Epoch summary statistics
    d_update_ratio = d_update_counter / len(train_loader)
    g_success_ratio = g_better_counter / len(train_loader)

    print(f"\n{'=' * 80}")
    print(f"📈 Epoch [{epoch + 1}/200] Summary:")
    print(f"{'=' * 80}")
    print(f"Content Loss:          {losses_c.avg:.6f}")
    print(f"Adversarial Loss (G):  {losses_a_g.avg:.6f}")
    print(f"Total G Loss:          {losses_g.avg:.6f}")
    print(f"Discriminator Loss:    {losses_d.avg:.6f}")
    print(f"D Real Accuracy:       {d_real_acc.avg:.2%}")
    print(f"D Fake Accuracy:       {d_fake_acc.avg:.2%}")
    print(f"D Update Rate:         {d_update_ratio:.2%}")
    print(f"G Success Rate:        {g_success_ratio:.2%}")
    print(f"Loss Ratio (C/A):      {losses_c.avg / (losses_a_g.avg + 1e-8):.2f}")
    print(f"{'=' * 80}\n")

    # Memory cleanup
    del lr_imgs, hr_imgs, sr_imgs, sr_imgs_norm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "loss_d": losses_d.avg,
        "loss_content": losses_c.avg,
        "loss_adversarial": losses_a_g.avg,
        "loss_g_total": losses_g.avg,
        "d_real_acc": d_real_acc.avg,
        "d_fake_acc": d_fake_acc.avg,
        "d_update_ratio": d_update_ratio,
        "g_success_ratio": g_success_ratio,
    }