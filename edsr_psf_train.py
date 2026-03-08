import os
import math
import re
import json
import time
import random
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import multi-band configuration files
from config import (
    init_multiband_config, SUPPORTED_WAVELENGTHS, DegradationConfig as DC
)

# Import split modules (key adaptation: remove gradient_loss/flux_loss imports, retain only validation metrics)
from dataset_loader import (
    HRtoLRDataset, load_psf_fits, crop_psf_by_energy, normalize_psf_kernel,
    psf_kernel_to_conv_weights, synthesize_lr_from_hr
)
from loss_functions import (
    psnr_torch, ssim_fallback, calculate_flux_error, calculate_spectral_ratio,
)
from visualization_utils import visualize_train_step, plot_training_curve
from edsr_models import EDSR as EDSR_Lite

# Utility functions (unchanged)
def count_model_params(model, verbose=True):
    """Calculate parameter count of PyTorch model"""
    if isinstance(model, nn.DataParallel):
        model = model.module
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def format_params(num):
        if num >= 1e6:
            return f"{num/1e6:.2f} M"
        elif num >= 1e3:
            return f"{num/1e3:.2f} K"
        else:
            return f"{num} "
    
    if verbose:
        print(f"\n========== Model Parameter Statistics ==========")
        print(f"Total Parameters: {format_params(total_params)} ({total_params:,})")
        print(f"Trainable Parameters: {format_params(trainable_params)} ({trainable_params:,})")
        print(f"Trainable Ratio: {trainable_params/total_params*100:.2f}%")
        print(f"===============================================\n")
    return total_params, trainable_params

def format_time(seconds):
    """Format time to hh:mm:ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:.2f}"
    else:
        return f"{minutes:02d}:{secs:.2f}"

def worker_init_fn(worker_id):
    """Fix random seed for DataLoader multi-threading"""
    seed = train_cfg.SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)

def numpy_json_serializer(obj):
    """Handle numpy type JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Unserializable type: {type(obj)}")

# Validation function (unchanged, retain all validation metrics including flux/spectral ratio)
def validate(model, val_loader, scale, device, train_cfg, psf_kernel_tensor, global_min, global_max):
    """Validate model (adapted for multi-band config) - use wavelength-specific hard clipping threshold, retain all validation metrics"""
    model.eval()
    metrics = {
        "lr_reproj_L1": 0.0, "psnr": 0.0, "ssim": 0.0,
        "flux_error": 0.0, "spectral_ratio": 0.0, "count": 0
    }
    vis_lr, vis_hr, vis_sr = None, None, None
    
    with torch.no_grad():
        for i, (lr, hr, _) in enumerate(val_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            if lr.ndim == 3:
                lr = lr.unsqueeze(1)
            if hr.ndim == 3:
                hr = hr.unsqueeze(1)
            
            sr = model(lr)
            sr = sr.clip(min=global_min, max=global_max)
            
            gen_lr = synthesize_lr_from_hr(sr, psf_kernel_tensor, scale, device)
            
            if i == train_cfg.VIS_SAMPLE_IDX and vis_lr is None:
                vis_lr = lr[0].detach().cpu()
                vis_hr = hr[0].detach().cpu()
                vis_sr = sr[0].detach().cpu()
            
            metrics["lr_reproj_L1"] += F.l1_loss(gen_lr, lr).item()
            metrics["psnr"] += psnr_torch(sr, hr)
            
            try:
                from metrics import torch_ssim, log_normalize
                metrics["ssim"] += torch_ssim(log_normalize(hr), log_normalize(sr)).mean().item()
            except:
                metrics["ssim"] += ssim_fallback(sr, hr, data_range=float(hr.max()-hr.min()))

            flux_err = calculate_flux_error(sr, hr)
            metrics["flux_error"] += flux_err
            metrics["spectral_ratio"] += calculate_spectral_ratio(sr, hr)
            metrics["count"] += 1

    if metrics["count"] == 0:
        return {}, None, None, None
    avg_metrics = {k: v/metrics["count"] for k, v in metrics.items() if k != "count"}
    
    print(f"\n📊 {train_cfg.WAVELENGTH}Å Validation Set Flux Summary:")
    print(f"   Total Samples: {metrics['count']} | Average Flux Error: {avg_metrics['flux_error']:.2f}%")
    
    return avg_metrics, vis_lr, vis_hr, vis_sr

# Training function (core adaptation: remove gradient/flux loss, retain only LR reprojection L1)
def train_single_band(dc, data_cfg, psf_cfg, train_cfg, infer_cfg):
    """
    Train EDSR model for single wavelength (adapted: only retain LR reprojection L1 loss, keep all validation metrics)
    :param dc: DegradationConfig instance (current wavelength)
    :param data_cfg: DataConfig instance (current wavelength)
    :param psf_cfg: PSFConfig instance (current wavelength)
    :param train_cfg: TrainConfig instance (current wavelength)
    :param infer_cfg: InferenceConfig instance (current wavelength)
    """
    # Explicitly specify using second GPU (cuda:1)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
        torch.cuda.set_device(1)
        print(f"\n✅ Switched to GPU1 (cuda:1) for training")
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        device = torch.device("cuda:0")
        print(f"\n⚠️  Only GPU0 is available, automatically fallback to GPU0 for training")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  No GPU available, using CPU for training (extremely slow)")

    # Fix random seed for current wavelength (paper-level reproducibility)
    seed = train_cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # Print training configuration for current wavelength (adapted: only show LR reprojection loss)
    print(f"\n" + "="*80)
    print(f"🚀 Start Training AIA {dc.AIA_WAVELENGTH}Å Band (Exclusive Configuration)")
    print(f"="*80)
    print(f"Device: {device}")
    print(f"Current GPU ID: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    print(f"Random Seed: {seed} (Globally fixed, perfect reproducibility)")
    print(f"HR Directory: {dc.HR_FITS_DIR}")
    print(f"Exclusive PSF Path: {dc.PSF_PATH}")
    print(f"PSF Energy Retention Ratio: {dc.PSF_ENERGY_CROP*100:.1f}%")
    print(f"Super-Resolution Scale: {dc.SCALE}×")
    print(f"Normalization Range: [{dc.GLOBAL_MIN}, {dc.GLOBAL_MAX}] DN")
    print(f"Training Epochs: {train_cfg.EPOCHS}")
    print(f"Batch Size: {train_cfg.BATCH_SIZE}")
    print(f"Model Output Directory: {train_cfg.OUT_DIR}")
    print(f"Visualization Output Directory: {train_cfg.VIS_OUT_DIR}")
    print(f"="*80)

    # Load exclusive PSF kernel for current wavelength
    print(f"\nReading {dc.AIA_WAVELENGTH}Å Exclusive PSF FITS: {dc.PSF_PATH}")
    psf_original = load_psf_fits(dc.PSF_PATH)
    print(f"{dc.AIA_WAVELENGTH}Å Original PSF Size: {psf_original.shape[0]}×{psf_original.shape[1]}")
    
    psf_original = normalize_psf_kernel(psf_original)
    psf_cropped = crop_psf_by_energy(psf_original, energy_ratio=dc.PSF_ENERGY_CROP, max_size=dc.PSF_MAX_SIZE)
    psf_kernel = normalize_psf_kernel(psf_cropped)
    print(f"{dc.AIA_WAVELENGTH}Å Final Cropped PSF: Size={psf_kernel.shape}, Energy Sum={psf_kernel.sum():.4f}")

    psf_kernel_tensor = psf_kernel_to_conv_weights(psf_kernel, channels=1, device=device)

    # Load HR files for current wavelength (path isolation)
    hr_paths = sorted([Path(dc.HR_FITS_DIR) / p for p in os.listdir(dc.HR_FITS_DIR) 
                      if p[0] != '.' and p.endswith(('.fits', '.fit'))])
    if len(hr_paths) == 0:
        raise RuntimeError(f"No FITS files found for {dc.AIA_WAVELENGTH}Å in {dc.HR_FITS_DIR}!")
    
    # Split train/validation set for current wavelength
    val_split = int(len(hr_paths) * dc.VAL_RATIO)
    train_paths = hr_paths[:-val_split] if val_split > 0 else hr_paths
    val_paths = hr_paths[-val_split:] if val_split > 0 else hr_paths
    
    # Build dataset for current wavelength (use wavelength-specific config)
    train_dataset = HRtoLRDataset(train_cfg, train_paths, psf_kernel if train_cfg.USE_PSF_IN_DATASET else None)
    val_dataset = HRtoLRDataset(train_cfg, val_paths, psf_kernel if train_cfg.USE_PSF_IN_DATASET else None)
    
    # Build DataLoader (fix seed for current wavelength)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True,
                              num_workers=train_cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn,
                              generator=torch.Generator().manual_seed(train_cfg.SEED))
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Initialize EDSR model for current wavelength (independent parameters)
    model = EDSR_Lite(n_resblocks=train_cfg.N_RESBLOCKS, n_feats=train_cfg.N_FEATS, scale=dc.SCALE)
    model = model.to(device)
    
    # DataParallel with GPU1
    if train_cfg.DATA_PARALLEL and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[1])
        print(f"{dc.AIA_WAVELENGTH}Å Band using GPU1 for DataParallel training")
    elif train_cfg.DATA_PARALLEL:
        print(f"\n⚠️ {dc.AIA_WAVELENGTH}Å Band: DATA_PARALLEL enabled but only 1 GPU available, auto disable multi-GPU parallelism")

    # Count model parameters for current wavelength
    count_model_params(model)

    # Optimizer (current wavelength parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.LR, 
        weight_decay=train_cfg.WEIGHT_DECAY
    )
    total_steps = train_cfg.EPOCHS * len(train_loader)

    # Learning rate scheduler (current wavelength parameters)
    if train_cfg.SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=train_cfg.MIN_LR
        )
    elif train_cfg.SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=train_cfg.STEP_SIZE,
            gamma=train_cfg.STEP_GAMMA
        )
    else:
        scheduler = None

    # Resume training (current wavelength model path, adapted: remove weight manager state loading)
    start_epoch = 0
    if train_cfg.RESUME and os.path.exists(train_cfg.RESUME):
        ck = torch.load(train_cfg.RESUME, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f"{dc.AIA_WAVELENGTH}Å Resume training from {train_cfg.RESUME}, start epoch: {start_epoch}")

    # Mixed precision training (current wavelength config)
    if train_cfg.AMP and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.AMP)
    else:
        scaler = torch.amp.GradScaler(enabled=False)

    # Training preparation (output directory auto-created for current wavelength)
    best_val_l1 = float('inf')
    metrics_history = []
    total_train_time = 0.0  # Total training time for current wavelength

    # Training loop (core adaptation: only retain LR reprojection loss calculation)
    for epoch in range(start_epoch, train_cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        running_loss_lr = 0.0  # Only retain LR reprojection loss accumulation

        epoch_start_time = time.time()

        for i, (lr_t, hr_t, _) in enumerate(train_loader):
            if lr_t.ndim == 3:
                lr_t = lr_t.unsqueeze(1)
            if hr_t.ndim == 3:
                hr_t = hr_t.unsqueeze(1)
            lr_t = lr_t.to(device)
            hr_t = hr_t.to(device)
            optimizer.zero_grad()

            # Forward pass (mixed precision)
            with torch.amp.autocast('cuda', enabled=train_cfg.AMP):
                sr_pred = model(lr_t)
                sr_pred = sr_pred.clip(min=dc.GLOBAL_MIN, max=dc.GLOBAL_MAX)
                
                gen_lr = synthesize_lr_from_hr(sr_pred, psf_kernel_tensor, dc.SCALE, device)
                
                # Core adaptation: only calculate LR reprojection L1 loss (only loss)
                loss_lr = F.l1_loss(gen_lr, lr_t)
                total_loss = loss_lr

            # Backward pass
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()
            scaler.update()

            # Log accumulation
            running_loss += total_loss.item()
            running_loss_lr += loss_lr.item()

            # Log printing
            if (i + 1) % train_cfg.LOG_INTERVAL == 0:
                avg_loss = running_loss / train_cfg.LOG_INTERVAL
                avg_loss_lr = running_loss_lr / train_cfg.LOG_INTERVAL
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"[{dc.AIA_WAVELENGTH}Å] [Epoch {epoch+1}/{train_cfg.EPOCHS}] Step {i+1}/{len(train_loader)} "
                      f"LR: {current_lr:.2e} | Total Loss: {avg_loss:.6f} | "
                      f"LR Reproj L1: {avg_loss_lr:.6f} )")
                
                running_loss = 0.0
                running_loss_lr = 0.0

        # Calculate epoch training time
        epoch_train_time = time.time() - epoch_start_time
        total_train_time += epoch_train_time
        print(f"\n✅ [{dc.AIA_WAVELENGTH}Å] Epoch {epoch+1} Training Completed | Time: {format_time(epoch_train_time)}")

        # Validation and visualization (retain all validation metrics)
        if (epoch + 1) % train_cfg.VAL_INTERVAL == 0 or epoch == train_cfg.EPOCHS - 1:
            val_metrics, vis_lr, vis_hr, vis_sr = validate(
                model, val_loader, dc.SCALE, device, train_cfg, psf_kernel_tensor,
                global_min=dc.GLOBAL_MIN,
                global_max=dc.GLOBAL_MAX
            )
            print(f"\n===== [{dc.AIA_WAVELENGTH}Å] Validation Epoch {epoch+1} =====")
            print(f"📊 Core Loss: LR Reprojection L1 = {val_metrics.get('lr_reproj_L1', 0):.6f}")
            print(f"📈 Image Metrics: PSNR = {val_metrics.get('psnr', 0):.3f} | SSIM = {val_metrics.get('ssim', 0):.4f}")
            print(f"🔭 Physical Metrics: Flux Error = {val_metrics.get('flux_error', 0)*100:.2f}% | Spectral Ratio = {val_metrics.get('spectral_ratio', 0):.4f}")
            print(f"=====================================================\n")

            # Save metrics history
            metrics_history.append({
                'epoch': epoch+1,
                'metrics': val_metrics,
                'wavelength': dc.AIA_WAVELENGTH
            })
            
            # Generate visualization results for current wavelength
            if vis_lr is not None and vis_hr is not None and vis_sr is not None:
                visualize_train_step(epoch+1, vis_lr, vis_hr, vis_sr, val_metrics, train_cfg)
            
            # Plot training curve for current wavelength
            plot_training_curve(metrics_history, train_cfg)

            # Save best model for current wavelength
            if val_metrics.get('lr_reproj_L1', float('inf')) < best_val_l1:
                best_val_l1 = val_metrics['lr_reproj_L1']
                ck_path = os.path.join(train_cfg.OUT_DIR, f"best_epoch_{epoch+1}_{dc.AIA_WAVELENGTH}Å.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics if 'val_metrics' in locals() else {},
                    'wavelength': dc.AIA_WAVELENGTH,
                    'cfg': train_cfg.__dict__
                }, ck_path)
                print(f"[{dc.AIA_WAVELENGTH}Å] Save best model to: {ck_path}\n")

        # Save checkpoint for current wavelength
        if (epoch + 1) % train_cfg.CKPT_INTERVAL == 0 or epoch == train_cfg.EPOCHS - 1:
            ck_path = os.path.join(train_cfg.OUT_DIR, f"epoch_{epoch+1}_{dc.AIA_WAVELENGTH}Å.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics if 'val_metrics' in locals() else {},
                'wavelength': dc.AIA_WAVELENGTH,
                'cfg': train_cfg.__dict__
            }, ck_path)
            print(f"[{dc.AIA_WAVELENGTH}Å] Save checkpoint to: {ck_path}\n")

    # Save complete metrics history for current wavelength
    metrics_history_path = os.path.join(train_cfg.VIS_OUT_DIR, f"full_metrics_history_{dc.AIA_WAVELENGTH}Å.json")
    with open(metrics_history_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_history, f, indent=4, default=numpy_json_serializer)
    print(f"[{dc.AIA_WAVELENGTH}Å] Complete training metrics history saved to: {metrics_history_path}")

    # Print training time summary for current wavelength
    print(f"\n========== [{dc.AIA_WAVELENGTH}Å] Training Time Summary ==========")
    print(f"Total Training Epochs: {train_cfg.EPOCHS - start_epoch} Epoch")
    print(f"Total Training Time: {format_time(total_train_time)}")
    print(f"Average Time per Epoch: {format_time(total_train_time/(train_cfg.EPOCHS - start_epoch))}")
    print(f"==================================================================")

    print(f"\n🎉 [{dc.AIA_WAVELENGTH}Å] Band Training Completed!")
    print(f"   Best Model Path: {train_cfg.OUT_DIR}")
    print(f"   Visualization Results Path: {train_cfg.VIS_OUT_DIR}")

    # Clear GPU1 cache after training each wavelength
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 [{dc.AIA_WAVELENGTH}Å] GPU1 cache cleared")

# Multi-band auto-training main function (unchanged)
def train_multiband(target_wavelengths=None):
    """
    Multi-band auto-training entry point (unchanged)
    :param target_wavelengths: Specified wavelength list for training (train all supported wavelengths if None)
    """
    # Check GPU status first
    if torch.cuda.is_available():
        print(f"\n📋 GPU Status Check:")
        print(f"   Available GPUs: {torch.cuda.device_count()}")
        print(f"   GPU1 Name: {torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else 'N/A'}")
    else:
        print(f"\n⚠️  No available GPU detected, will use CPU for training")

    # Determine wavelength list to train
    train_wavelengths = target_wavelengths if target_wavelengths else SUPPORTED_WAVELENGTHS
    
    print(f"\n" + "="*100)
    print(f"📡 Start EDSR Multi-band Auto-training | Wavelengths to Train: {train_wavelengths} Å")
    print(f"   Training Device: GPU1 (cuda:1)")
    print(f"="*100)

    # Iterate training for each wavelength
    for idx, wavelength in enumerate(train_wavelengths):
        try:
            # 1. Initialize exclusive config for current wavelength
            band_configs = init_multiband_config(wavelength=wavelength)
            
            # 2. Extract various configs for current wavelength
            dc = band_configs["degradation"]
            data_cfg = band_configs["data"]
            psf_cfg = band_configs["psf"]
            train_cfg = band_configs["train"]
            infer_cfg = band_configs["inference"]
            
            # 3. Train current wavelength
            train_cfg.WAVELENGTH = dc.AIA_WAVELENGTH
            train_single_band(dc, data_cfg, psf_cfg, train_cfg, infer_cfg)
            
            # 4. Wavelength switch prompt
            if idx < len(train_wavelengths) - 1:
                next_wave = train_wavelengths[idx+1]
                print(f"\n" + "="*100)
                print(f"🔄 [{wavelength}Å] Training Completed, switching to [{next_wave}Å] band training...")
                print(f"="*100)
                torch.cuda.empty_cache()
                time.sleep(2)
        
        except Exception as e:
            print(f"\n❌ [{wavelength}Å] Band Training Error: {str(e)}")
            print(f"⚠️  Skip current wavelength, continue training next wavelength...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # All wavelengths training completed
    print(f"\n" + "="*100)
    print(f"🏆 All Specified Wavelengths Training Completed!")
    print(f"   Trained Wavelengths: {train_wavelengths} Å")
    print(f"   Training Device: GPU1 (cuda:1)")
    print(f"   All models are saved in their respective wavelength-exclusive directories")
    print(f"="*100)

# Run entry (unchanged)
if __name__ == '__main__':
    # Method 1: Train all supported wavelengths (94/131/171/193/211/304/335)
    # train_multiband()
    
    # Method 2: Train specified wavelengths (e.g., only 171Å)
    train_multiband(target_wavelengths=[171])
