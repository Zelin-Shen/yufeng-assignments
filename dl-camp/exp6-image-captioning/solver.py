import torch
import torch.nn as nn 
import time 
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from utils import *


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, cfg, visualizer=None):
   """
   Train for one epoch with visualization support
   
   Args:
       train_loader: Training data loader
       encoder: Encoder model
       decoder: Decoder model
       criterion: Loss function
       encoder_optimizer: Encoder optimizer
       decoder_optimizer: Decoder optimizer
       epoch: Current epoch number
       cfg: Configuration dictionary
       visualizer: Visualizer object for tracking metrics
   """
   decoder.train()
   encoder.train()
   
   batch_time = AverageMeter()
   data_time = AverageMeter()
   losses = AverageMeter()
   top5accs = AverageMeter()
   
   # Collect scalar attention statistics (avoids dimension mismatch)
   alpha_stats = []  # Store mean alpha value per sample
   beta_stats = []   # Store mean beta value per sample
   
   start = time.time()

   for i, (imgs, caps, caplens) in enumerate(train_loader):
       data_time.update(time.time() - start)
       
       imgs = imgs.to(cfg['device'])
       caps = caps.to(cfg['device'])
       caplens = caplens.to(cfg['device'])

       # Forward pass
       imgs = encoder(imgs)
       decoder_outputs = decoder(imgs, caps, caplens)
       
       # Unpack decoder outputs based on type
       if len(decoder_outputs) == 4:  # DecoderWithRNN (no attention)
           scores, caps_sorted, decode_lengths, sort_ind = decoder_outputs
           alphas = None
           betas = None
       elif len(decoder_outputs) == 5:  # DecoderWithAttention
           scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_outputs
           betas = None
       elif len(decoder_outputs) == 6:  # DecoderWithAdaptiveAttention
           scores, caps_sorted, decode_lengths, alphas, betas, sort_ind = decoder_outputs
       else:
           raise ValueError(f"Unexpected decoder outputs: {len(decoder_outputs)}")

       # Collect attention statistics: compute mean per sample to avoid shape mismatch
       if alphas is not None:
           # alphas: (batch_size, max_decode_length, num_pixels)
           # Compute mean across spatial and temporal dims -> (batch_size,)
           batch_alpha_means = alphas.mean(dim=(1, 2)).cpu().detach().numpy()
           alpha_stats.extend(batch_alpha_means.tolist())
       
       if betas is not None:
           # betas: (batch_size, max_decode_length, 1)
           # Compute mean across temporal dim -> (batch_size,)
           batch_beta_means = betas.mean(dim=1).squeeze(-1).cpu().detach().numpy()
           beta_stats.extend(batch_beta_means.tolist())

       # Targets are all words after <start>, up to <end>
       targets = caps_sorted[:, 1:]

       # Remove timesteps that weren't decoded or pads
       scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
       targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

       # Compute loss
       loss = criterion(scores, targets)

       # Add doubly stochastic attention regularization if applicable
       if cfg.get('attention', False) and alphas is not None:
           loss += cfg.get('alpha_c', 1.0) * ((1. - alphas.sum(dim=1)) ** 2).mean()

       # Backward pass
       decoder_optimizer.zero_grad()
       if encoder_optimizer is not None:
           encoder_optimizer.zero_grad()
       loss.backward()

       # Gradient clipping
       if cfg.get('grad_clip') is not None:
           clip_gradient(decoder_optimizer, cfg['grad_clip'])
           if encoder_optimizer is not None:
               clip_gradient(encoder_optimizer, cfg['grad_clip'])

       # Update weights
       decoder_optimizer.step()
       if encoder_optimizer is not None:
           encoder_optimizer.step()

       # Track metrics
       top5 = accuracy(scores, targets, 5)
       losses.update(loss.item(), sum(decode_lengths))
       top5accs.update(top5, sum(decode_lengths))
       batch_time.update(time.time() - start)

       start = time.time()

       # Print status
       if i % cfg.get('print_freq', 100) == 0:
           print('Epoch: [{0}][{1}/{2}]\t'
                 'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                 'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                 'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                     epoch, i, len(train_loader),
                     batch_time=batch_time,
                     data_time=data_time, 
                     loss=losses,
                     top5=top5accs))
   
   # Update visualizer with epoch metrics
   if visualizer is not None:
       # Update training metrics
       visualizer.update_train_metrics(
           epoch=epoch,
           loss=losses.avg,
           top5_acc=top5accs.avg,
           batch_time=batch_time.avg,
           data_time=data_time.avg
       )
       
       # Update learning rate info
       encoder_lr = encoder_optimizer.param_groups[0]['lr'] if encoder_optimizer is not None else None
       decoder_lr = decoder_optimizer.param_groups[0]['lr']
       visualizer.update_lr_metrics(epoch, encoder_lr, decoder_lr)
       
       # Update attention statistics: convert list to numpy array (now 1D, safe)
       if len(alpha_stats) > 0:
           alpha_array = np.array(alpha_stats)
           alpha_mean = float(np.mean(alpha_array))
           alpha_std = float(np.std(alpha_array))
           
           beta_mean = beta_std = None
           if len(beta_stats) > 0:
               beta_array = np.array(beta_stats)
               beta_mean = float(np.mean(beta_array))
               beta_std = float(np.std(beta_array))
           
           visualizer.update_attention_metrics(epoch, alpha_mean, alpha_std, beta_mean, beta_std)
       
       print(f"✓ Training metrics updated for epoch {epoch}")


def validate(val_loader, encoder, decoder, criterion, word_map, cfg, visualizer=None):
   """
   Validate for one epoch with visualization support
   
   Args:
       val_loader: Validation data loader
       encoder: Encoder model
       decoder: Decoder model
       criterion: Loss function
       word_map: Word-to-index mapping
       cfg: Configuration dictionary
       visualizer: Visualizer object for tracking metrics
       
   Returns:
       bleu4: BLEU-4 score
   """
   decoder.eval()
   if encoder is not None:
       encoder.eval()

   batch_time = AverageMeter()
   losses = AverageMeter()
   top5accs = AverageMeter()
   
   # Collect scalar attention statistics
   alpha_stats = []
   beta_stats = []

   start = time.time()

   references = list()  # Ground truth captions for BLEU-4 calculation
   hypotheses = list()  # Predicted captions

   # Disable gradients to save memory
   with torch.no_grad():
       for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

           # Move to device
           imgs = imgs.to(cfg['device'])
           caps = caps.to(cfg['device'])
           caplens = caplens.to(cfg['device'])

           # Forward pass
           if encoder is not None:
               imgs = encoder(imgs)
           
           decoder_outputs = decoder(imgs, caps, caplens)
           
           # Unpack decoder outputs
           if len(decoder_outputs) == 4:  # DecoderWithRNN
               scores, caps_sorted, decode_lengths, sort_ind = decoder_outputs
               alphas = None
               betas = None
           elif len(decoder_outputs) == 5:  # DecoderWithAttention
               scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder_outputs
               betas = None
           elif len(decoder_outputs) == 6:  # DecoderWithAdaptiveAttention
               scores, caps_sorted, decode_lengths, alphas, betas, sort_ind = decoder_outputs
           else:
               raise ValueError(f"Unexpected decoder outputs: {len(decoder_outputs)}")

           # Collect attention statistics: mean per sample
           if alphas is not None:
               batch_alpha_means = alphas.mean(dim=(1, 2)).cpu().numpy()
               alpha_stats.extend(batch_alpha_means.tolist())
           
           if betas is not None:
               batch_beta_means = betas.mean(dim=1).squeeze(-1).cpu().numpy()
               beta_stats.extend(batch_beta_means.tolist())

           # Targets are words after <start>
           targets = caps_sorted[:, 1:]

           # Remove pads and undecodable timesteps
           scores_copy = scores.clone()
           scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
           targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

           # Compute loss
           loss = criterion(scores, targets)

           # Add doubly stochastic attention regularization
           if cfg.get('attention', False) and alphas is not None:
               loss += cfg.get('alpha_c', 1.0) * ((1. - alphas.sum(dim=1)) ** 2).mean()

           # Track metrics
           losses.update(loss.item(), sum(decode_lengths))
           top5 = accuracy(scores, targets, 5)
           top5accs.update(top5, sum(decode_lengths))
           batch_time.update(time.time() - start)

           start = time.time()

           if i % cfg.get('print_freq', 100) == 0:
               print('Validation: [{0}/{1}]\t'
                     'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                         i, len(val_loader), 
                         batch_time=batch_time,
                         loss=losses, 
                         top5=top5accs))

           # Store references and hypotheses for BLEU computation
           # References format: [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...]
           # Hypotheses format: [hyp1, hyp2, ...]
           allcaps = allcaps[sort_ind.cpu()]  # Images were sorted in decoder
           for j in range(allcaps.shape[0]):
               img_caps = allcaps[j].tolist()
               img_captions = list(
                   map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                       img_caps))  # Remove <start> and pads
               references.append(img_captions)

           # Hypotheses
           _, preds = torch.max(scores_copy, dim=2)
           preds = preds.tolist()
           temp_preds = list()
           for j, p in enumerate(preds):
               temp_preds.append(preds[j][:decode_lengths[j]])  # Remove pads
           preds = temp_preds
           hypotheses.extend(preds)

           assert len(references) == len(hypotheses)

       # Compute BLEU-4 score
       bleu4 = corpus_bleu(references, hypotheses)

       print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
           loss=losses,
           top5=top5accs,
           bleu=bleu4))
   
   # Update visualizer with validation metrics
   if visualizer is not None:
       visualizer.update_val_metrics(
           epoch=cfg.get('current_epoch', 0),
           loss=losses.avg,
           top5_acc=top5accs.avg,
           bleu4=bleu4
       )
       
       print(f"✓ Validation metrics updated")

   return bleu4


def train_with_visualization(train_loader, val_loader, encoder, decoder, criterion, 
                            encoder_optimizer, decoder_optimizer, word_map, cfg, visualizer):
   """
   Complete training loop with visualization and checkpoint saving
   
   Args:
       train_loader: Training data loader
       val_loader: Validation data loader
       encoder: Encoder model
       decoder: Decoder model
       criterion: Loss function
       encoder_optimizer: Encoder optimizer
       decoder_optimizer: Decoder optimizer
       word_map: Word-to-index mapping
       cfg: Configuration dictionary
       visualizer: Visualizer object
       
   Returns:
       best_bleu4: Best BLEU-4 score achieved
   """
   print(f"\n{'='*70}")
   print(f"Starting Training - {visualizer.experiment_name}")
   print(f"{'='*70}\n")
   
   # Save configuration
   visualizer.save_config(cfg)
   
   best_bleu4 = 0.
   epochs_since_improvement = 0
   
   for epoch in range(cfg.get('start_epoch', 0), cfg['epochs']):
       # Save current epoch for visualization
       cfg['current_epoch'] = epoch
       
       # Early stopping check
       if epochs_since_improvement == 20:
           print("\n⚠ No improvement for 20 epochs. Stopping training.\n")
           break
           
       # Learning rate decay
       if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
           print(f"\n⚡ Adjusting learning rate (no improvement for {epochs_since_improvement} epochs)")
           adjust_learning_rate(decoder_optimizer, 0.8)
           if cfg.get('fine_tune_encoder', False) and encoder_optimizer is not None:
               adjust_learning_rate(encoder_optimizer, 0.8)

       # Train for one epoch
       print(f"\n{'─'*70}")
       print(f"Epoch {epoch}/{cfg['epochs']-1} - Training")
       print(f"{'─'*70}")
       train(
           train_loader=train_loader,
           encoder=encoder,
           decoder=decoder,
           criterion=criterion,
           encoder_optimizer=encoder_optimizer,
           decoder_optimizer=decoder_optimizer,
           epoch=epoch,
           cfg=cfg,
           visualizer=visualizer
       )

       # Validate for one epoch
       print(f"\n{'─'*70}")
       print(f"Epoch {epoch}/{cfg['epochs']-1} - Validation")
       print(f"{'─'*70}")
       recent_bleu4 = validate(
           val_loader=val_loader,
           encoder=encoder,
           decoder=decoder,
           criterion=criterion,
           word_map=word_map,
           cfg=cfg,
           visualizer=visualizer
       )

       # Check for improvement
       is_best = recent_bleu4 > best_bleu4
       best_bleu4 = max(recent_bleu4, best_bleu4)
       
       if not is_best:
           epochs_since_improvement += 1
           print(f"\n⚠ Epochs since last improvement: {epochs_since_improvement}")
       else:
           epochs_since_improvement = 0
           print(f"\n🎉 New best BLEU-4: {best_bleu4:.4f}!")

       # Save checkpoint
       save_checkpoint(
           cfg['data_name'],
           epoch,
           epochs_since_improvement,
           encoder,
           decoder,
           encoder_optimizer,
           decoder_optimizer,
           recent_bleu4,
           is_best,
       )
       
       # Generate visualizations after each epoch
       print(f"\n📊 Generating visualizations...")
       try:
           visualizer.plot_all()
           visualizer.save_metrics()
       except Exception as e:
           print(f"⚠ Warning: Visualization failed - {e}")
       
       # Generate report every 5 epochs
       if (epoch + 1) % 5 == 0:
           try:
               visualizer.generate_report()
           except Exception as e:
               print(f"⚠ Warning: Report generation failed - {e}")
   
   # Generate final report after training
   print(f"\n{'='*70}")
   print(f"Training Complete!")
   print(f"Best BLEU-4: {best_bleu4:.4f}")
   print(f"{'='*70}\n")
   
   try:
       visualizer.generate_report()
       print(f"\n✓ All results saved to: {visualizer.exp_dir}\n")
   except Exception as e:
       print(f"⚠ Warning: Final report generation failed - {e}")
   
   return best_bleu4