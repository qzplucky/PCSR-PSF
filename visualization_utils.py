#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import sunpy.map
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch
from scipy.ndimage import gaussian_filter, label, find_objects
from astropy.io import fits

WAVELENGTH_TO_CMAP = {
    94: 'sdoaia94',
    131: 'sdoaia131',
    171: 'sdoaia171',
    193: 'sdoaia193',
    211: 'sdoaia211',
    304: 'sdoaia304',
    335: 'sdoaia335'
}

AIA_OFFICIAL_CONFIG = {
    94: (30, 99.9, True),    
    131: (25, 99.9, True),   
    171: (0, 99.5, False),  
    193: (1, 99.5, False),  
    211: (1, 99.5, False),  
    304: (30, 99.9, True),   
    335: (30, 99.9, True)    
}
SUPPORTED_WAVELENGTHS = list(AIA_OFFICIAL_CONFIG.keys())
HIGH_NOISE_WAVELENGTHS = [94, 131, 335]
SMOOTH_SIGMA = 0
CROP_EXTEND = 100
MIN_FLARE_PIXELS = 50
DIFF_PERCENTILE = 99
ZOOM_REGION_DEFAULT = {'x1': 200, 'x2': 600, 'y1': 200, 'y2': 600}
MIN_ZOOM_SIZE = 400

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.figsize": (24, 12),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": 'tight',
    "savefig.pad_inches": 0.1,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def convert_numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(v) for v in obj]
    else:
        return obj

def mild_denoise_aia_data(data, wavelength):
    if data.size == 0 or np.all(np.isnan(data)):
        print(f"⚠️ Data is empty/all NaN before denoising (wavelength {wavelength}Å), skipping denoising")
        return data
    
    if wavelength in HIGH_NOISE_WAVELENGTHS:
        denoised_data = np.copy(data)
        valid_mask = ~np.isnan(denoised_data)
        if np.sum(valid_mask) == 0:
            print(f"⚠️ No valid data for denoising (wavelength {wavelength}Å)")
            return denoised_data
        denoised_data[valid_mask] = gaussian_filter(denoised_data[valid_mask], sigma=SMOOTH_SIGMA)
        return denoised_data
    return data

def get_aia_visual_config(wavelength):
    wave_str = str(wavelength)
    pure_wave = ''.join([c for c in wave_str if c.isdigit()])
    pure_wave = int(pure_wave) if pure_wave else 171
    
    pure_wave = pure_wave if pure_wave in SUPPORTED_WAVELENGTHS else 171
    cmap_name = WAVELENGTH_TO_CMAP.get(pure_wave, 'sdoaia171')
    cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap('gray')
    min_pct, max_pct, use_log = AIA_OFFICIAL_CONFIG[pure_wave]
    return cmap, min_pct, max_pct, use_log

def get_percentile_limits(data, min_pct, max_pct):
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        print(f"⚠️ No valid data to calculate percentile limits, using default values (0,1)")
        return 0, 1
    vmin = np.percentile(valid_data, min_pct)
    vmax = np.percentile(valid_data, max_pct)
    if vmin == vmax:
        vmax = vmin + 1e-6 if vmin == 0 else vmin * 1.01
    return vmin, vmax

def auto_detect_flare(data, wavelength):
    if data.size == 0 or np.all(np.isnan(data)):
        print(f"⚠️ Flare detection failed: Data is empty/all NaN (wavelength {wavelength}Å)")
        return None
    
    data_denoise = mild_denoise_aia_data(data.copy(), wavelength)
    _, min_pct, max_pct, _ = get_aia_visual_config(wavelength)
    valid_data = data_denoise[~np.isnan(data_denoise)]
    if len(valid_data) == 0:
        print(f"⚠️ Flare detection failed: No valid data (wavelength {wavelength}Å)")
        return None
    
    flare_thresh = np.percentile(valid_data, max_pct)
    flare_mask = data_denoise >= flare_thresh
    labeled_mask, num_features = label(flare_mask)
    
    if num_features == 0:
        print(f"⚠️ Flare detection failed: No bright regions detected (wavelength {wavelength}Å)")
        return None
    
    flare_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features+1)]
    max_flare_idx = np.argmax(flare_sizes) + 1
    max_flare_size = flare_sizes[max_flare_idx-1]
    
    if max_flare_size < MIN_FLARE_PIXELS:
        print(f"⚠️ Flare detection failed: Maximum bright region has {max_flare_size} pixels (<{MIN_FLARE_PIXELS}), identified as noise")
        return None
    
    flare_slices = find_objects(labeled_mask == max_flare_idx)[0]
    x1, y1 = flare_slices[1].start, flare_slices[0].start
    x2, y2 = flare_slices[1].stop, flare_slices[0].stop
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

def get_crop_coords(flare_region, data_shape, extend):
    height, width = data_shape
    x1, y1 = flare_region['x1'], flare_region['y1']
    x2, y2 = flare_region['x2'], flare_region['y2']
    
    x1 = max(0, x1 - extend)
    y1 = max(0, y1 - extend)
    x2 = min(width, x2 + extend)
    y2 = min(height, y2 + extend)
    
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

def get_single_diff_normalize(diff_data, percentile):
    valid_diff = diff_data[~np.isnan(diff_data)]
    if len(valid_diff) == 0:
        print(f"⚠️ No valid data for difference map, using default thresholds")
        return plt.get_cmap('seismic'), ImageNormalize(vmin=-1, vmax=1, stretch=LinearStretch()), -1, 1
    
    abs_max = np.percentile(np.abs(valid_diff), percentile)
    vmin = -abs_max
    vmax = abs_max
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    cmap = plt.get_cmap('seismic')
    return cmap, norm, vmin, vmax

def save_aia_fits(data, save_path, wavelength, epoch, data_type):
    try:
        if data.size == 0 or np.all(np.isnan(data)):
            print(f"⚠️ Skipping empty data FITS save: {save_path}")
            return False
        
        fits_data = np.nan_to_num(data).astype(np.float32)
        hdu = fits.PrimaryHDU(fits_data)
        wave_str = str(wavelength)
        pure_wave = ''.join([c for c in wave_str if c.isdigit()])
        pure_wave = int(pure_wave) if pure_wave else wavelength
        hdu.header['WAVE'] = (pure_wave, 'AIA wavelength (Angstrom)')
        hdu.header['EPOCH'] = (epoch, 'Training epoch')
        hdu.header['DATATYPE'] = (data_type, 'Data type (Degraded/Clean/Recovered)')
        hdu.header['CREATED'] = (np.datetime64('now').astype(str), 'File creation time')
        hdu.header['DATA_PER'] = (wave_str, 'Data period (2024 quiet/2021 flare)')
        
        hdu.header['MIN'] = (np.min(fits_data) if fits_data.size > 0 else 0, 'Minimum intensity value')
        hdu.header['MAX'] = (np.max(fits_data) if fits_data.size > 0 else 1, 'Maximum intensity value')
        hdu.header['MEAN'] = (np.mean(fits_data) if fits_data.size > 0 else 0.5, 'Mean intensity value')
        
        hdul = fits.HDUList([hdu])
        hdul.writeto(save_path, overwrite=True)
        hdul.close()
        return True
    except Exception as e:
        print(f"⚠️ FITS save failed {save_path}: {str(e)}")
        return False

def validate_zoom_region(zoom_region, data_shape):
    height, width = data_shape
    x1, x2 = zoom_region['x1'], zoom_region['x2']
    y1, y2 = zoom_region['y1'], zoom_region['y2']
    
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height))
    
    if x2 - x1 < MIN_ZOOM_SIZE:
        x2 = x1 + MIN_ZOOM_SIZE
        if x2 > width:
            x1 = max(0, width - MIN_ZOOM_SIZE)
            x2 = width
    if y2 - y1 < MIN_ZOOM_SIZE:
        y2 = y1 + MIN_ZOOM_SIZE
        if y2 > height:
            y1 = max(0, height - MIN_ZOOM_SIZE)
            y2 = height
    
    if x1 >= x2 or y1 >= y2:
        print(f"⚠️ Invalid zoom region coordinates, using default region")
        return ZOOM_REGION_DEFAULT.copy()
    
    valid_region = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
    return valid_region

def visualize_train_step(epoch, deg, clean, rec, metrics, cfg, lr=None, hr=None, sr=None):
    original_wave = getattr(cfg, 'AIA_WAVELENGTH', 171)
    wave_str = str(original_wave)
    period_configs = []
    if '2024' in wave_str:
        period_configs.append(('quiet_2024', original_wave, 'quiet_2024'))
    if '2021' in wave_str:
        period_configs.append(('flare_2021', original_wave, 'flare_2021'))
    if not period_configs:
        period_configs.append(('original', original_wave, ''))
    
    main_path = None
    for period_name, period_wave, save_suffix in period_configs:
        cfg.AIA_WAVELENGTH = period_wave
        
        try:
            aia_wavelength = cfg.AIA_WAVELENGTH
        except:
            aia_wavelength = 171
        
        deg = deg if deg is not None else lr
        clean = clean if clean is not None else hr
        rec = rec if rec is not None else sr
        deg_np = deg.squeeze().numpy() if torch.is_tensor(deg) else np.squeeze(deg)
        clean_np = clean.squeeze().numpy() if torch.is_tensor(clean) else np.squeeze(clean)
        rec_np = rec.squeeze().numpy() if torch.is_tensor(rec) else np.squeeze(rec)

        for name, data in zip(['Degraded', 'Clean', 'Recovered'], [deg_np, clean_np, rec_np]):
            if data.size == 0:
                raise ValueError(f"❌ {period_name} - {name} data is empty (wavelength {aia_wavelength}), cannot visualize")
            if len(data.shape) != 2:
                print(f"⚠️ {period_name} - {name} data has abnormal dimensions({data.shape}), forcing to 2D")
                data = data.reshape(-1, data.shape[-1]) if len(data.shape) > 2 else data
        
        crop_extend = getattr(cfg, 'CROP_EXTEND', CROP_EXTEND)
        flare_region = auto_detect_flare(clean_np, aia_wavelength)
        if flare_region is not None:
            zoom_region = get_crop_coords(flare_region, clean_np.shape, crop_extend)
        else:
            try:
                zoom_region = cfg.ZOOM_REGION
            except:
                zoom_region = ZOOM_REGION_DEFAULT.copy()
        zoom_region = validate_zoom_region(zoom_region, clean_np.shape)
        x1, x2, y1, y2 = zoom_region['x1'], zoom_region['x2'], zoom_region['y1'], zoom_region['y2']
        
        os.makedirs(cfg.VIS_OUT_DIR, exist_ok=True)
        epoch_dir = os.path.join(cfg.VIS_OUT_DIR, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        deg_np = mild_denoise_aia_data(deg_np, aia_wavelength)
        clean_np = mild_denoise_aia_data(clean_np, aia_wavelength)
        rec_np = mild_denoise_aia_data(rec_np, aia_wavelength)
        
        deg_zoom = deg_np[y1:y2, x1:x2]
        clean_zoom = clean_np[y1:y2, x1:x2]
        rec_zoom = rec_np[y1:y2, x1:x2]
        
        diff_np = rec_np - clean_np
        diff_zoom = rec_zoom - clean_zoom if clean_zoom.size > 0 and rec_zoom.size > 0 else np.array([[0]])
        diff_cmap, diff_norm, diff_vmin, diff_vmax = get_single_diff_normalize(diff_np, DIFF_PERCENTILE)
        diff_zoom_cmap, diff_zoom_norm, diff_zoom_vmin, diff_zoom_vmax = get_single_diff_normalize(diff_zoom, DIFF_PERCENTILE)
        
        def safe_reduction(arr, func, default=0.0):
            valid_arr = arr[~np.isnan(arr)] if arr.size > 0 else np.array([])
            return func(valid_arr) if valid_arr.size > 0 else default
        
        diff_mean = safe_reduction(diff_np, np.mean, 0.0)
        diff_std = safe_reduction(diff_np, np.std, 0.0)
        diff_abs_max = safe_reduction(np.abs(diff_np), np.max, 0.0)
        diff_mean_zoom = safe_reduction(diff_zoom, np.mean, 0.0)
        diff_std_zoom = safe_reduction(diff_zoom, np.std, 0.0)
        diff_abs_max_zoom = safe_reduction(np.abs(diff_zoom), np.max, 0.0)
        
        cmap, min_pct, max_pct, use_log = get_aia_visual_config(aia_wavelength)
        vmin, vmax = get_percentile_limits(clean_np, min_pct, max_pct)
        stretch = LogStretch() if use_log else LinearStretch()
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)
        
        fig = plt.figure(figsize=(24, 12), dpi=300)
        
        ax1 = plt.subplot(2, 4, 1)
        im1 = ax1.imshow(deg_np, cmap=cmap, norm=norm)
        ax1.set_title(f'Degraded Input (AIA {aia_wavelength} | {period_name} | PSF Blur)', fontweight='bold')
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar1.ax.tick_params(labelsize=7)
        
        ax2 = plt.subplot(2, 4, 2)
        im2 = ax2.imshow(clean_np, cmap=cmap, norm=norm)
        ax2.set_title(f'Clean Reference (AIA Observation)', fontweight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar2.ax.tick_params(labelsize=7)
        
        ax3 = plt.subplot(2, 4, 3)
        im3 = ax3.imshow(rec_np, cmap=cmap, norm=norm)
        ax3.set_title(f'Recovered Output (EDSR)', fontweight='bold')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar3.ax.tick_params(labelsize=7)
        
        ax4 = plt.subplot(2, 4, 4)
        im4 = ax4.imshow(diff_np, cmap=diff_cmap, norm=diff_norm)
        ax4.set_title(f'Difference (Rec - Clean) | Mean: {diff_mean:.4f} | Std: {diff_std:.4f}', fontweight='bold')
        ax4.axis('off')
        diff_ticks = np.linspace(diff_vmin, diff_vmax, 5) if diff_abs_max > 0 else [-1, -0.5, 0, 0.5, 1]
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.6, ticks=diff_ticks)
        cbar4.ax.tick_params(labelsize=7)
        cbar4.set_label('Intensity Difference', fontsize=8)
        
        ax5 = plt.subplot(2, 4, 5)
        im5 = ax5.imshow(clean_zoom, cmap=cmap, norm=norm)
        ax5.set_title(f'Clean Zoom (Flare Region: {x1}:{x2}, {y1}:{y2})', fontweight='bold')
        ax5.axis('off')
        cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar5.ax.tick_params(labelsize=7)
        
        ax6 = plt.subplot(2, 4, 6)
        im6 = ax6.imshow(rec_zoom, cmap=cmap, norm=norm)
        ax6.set_title(f'Rec Zoom (Flare Region: {x1}:{x2}, {y1}:{y2})', fontweight='bold')
        ax6.axis('off')
        cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar6.ax.tick_params(labelsize=7)
        
        ax7 = plt.subplot(2, 4, 7)
        im7 = ax7.imshow(diff_zoom, cmap=diff_zoom_cmap, norm=diff_zoom_norm)
        ax7.set_title(f'Flare Region Diff | Mean: {diff_mean_zoom:.4f} | Std: {diff_std_zoom:.4f}', fontweight='bold')
        ax7.axis('off')
        diff_ticks_zoom = np.linspace(diff_zoom_vmin, diff_zoom_vmax, 5) if diff_abs_max_zoom > 0 else [-1, -0.5, 0, 0.5, 1]
        cbar7 = plt.colorbar(im7, ax=ax7, shrink=0.6, ticks=diff_ticks_zoom)
        cbar7.ax.tick_params(labelsize=7)
        cbar7.set_label('Intensity Difference', fontsize=8)
        
        ax8 = plt.subplot(2, 4, 8)
        im8 = ax8.imshow(clean_np, cmap=cmap, norm=norm)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='white', facecolor='none')
        ax8.add_patch(rect)
        ax8.set_title(f'Clean with Flare Region ({x1}:{x2}, {y1}:{y2})', fontweight='bold')
        ax8.axis('off')
        cbar8 = plt.colorbar(im8, ax=ax8, shrink=0.6, ticks=np.linspace(vmin, vmax, 5))
        cbar8.ax.tick_params(labelsize=7)
        
        fig_profile = plt.figure(figsize=(12, 4), dpi=300)
        ax_profile = fig_profile.add_subplot(111)
        center_y = clean_np.shape[0] // 2
        x = np.arange(clean_np.shape[1])
        clean_profile = clean_np[center_y, :]
        rec_profile = rec_np[center_y, :]
        
        ax_profile.plot(x, clean_profile, label='Clean Reference', c='red', linewidth=2, alpha=0.8)
        ax_profile.plot(x, rec_profile, label='Recovered Output', c='blue', linewidth=2, alpha=0.8)
        ax_profile.set_title(f'Horizontal Profile (AIA {aia_wavelength} | {period_name})', fontweight='bold')
        ax_profile.set_xlabel('Pixel Position')
        ax_profile.set_ylabel('Intensity')
        ax_profile.legend(fontsize=8)
        ax_profile.grid(True, alpha=0.3)
        
        wave_num = ''.join([c for c in str(aia_wavelength) if c.isdigit()])
        save_suffix = f"_{save_suffix}" if save_suffix else ""
        main_path = os.path.join(epoch_dir, f"epoch_{epoch}_core_comparison_{wave_num}A{save_suffix}.png")
        plt.tight_layout(pad=1.0)
        fig.savefig(main_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        profile_path = os.path.join(epoch_dir, f"epoch_{epoch}_profile_{wave_num}A{save_suffix}.png")
        fig_profile.savefig(profile_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig_profile)
        
        core_metrics = {
            'epoch': epoch,
            'aia_wavelength': aia_wavelength,
            'data_period': period_name,
            'psf_match_l1': round(metrics.get('lr_reproj_L1', metrics.get('psf_match_l1', 0)), 6),
            'flux_error': round(metrics.get('flux_error', 0), 2),
            'pcc': round(metrics.get('pcc', 0), 4),
            'psnr': round(metrics.get('psnr', 0), 2),
            'ssim': round(metrics.get('ssim', 0), 4),
            'edge_preservation': round(metrics.get('edge_preservation', 0), 4),
            'flare_detected': flare_region is not None,
            'zoom_region': zoom_region,
            'zoom_diff_mean': round(diff_mean_zoom, 4),
            'zoom_diff_std': round(diff_std_zoom, 4)
        }
        metrics_path = os.path.join(epoch_dir, f"epoch_{epoch}_core_metrics_{wave_num}A{save_suffix}.json")
        with open(metrics_path, 'w') as f:
            json.dump(convert_numpy_to_python(core_metrics), f, indent=4)
        
        clean_fits_path = os.path.join(epoch_dir, f"epoch_{epoch}_clean_{wave_num}A{save_suffix}.fits")
        rec_fits_path = os.path.join(epoch_dir, f"epoch_{epoch}_rec_{wave_num}A{save_suffix}.fits")
        deg_fits_path = os.path.join(epoch_dir, f"epoch_{epoch}_deg_{wave_num}A{save_suffix}.fits")
        save_aia_fits(clean_np, clean_fits_path, aia_wavelength, epoch, 'Clean')
        save_aia_fits(rec_np, rec_fits_path, aia_wavelength, epoch, 'Recovered')
        save_aia_fits(deg_np, deg_fits_path, aia_wavelength, epoch, 'Degraded')

    cfg.AIA_WAVELENGTH = original_wave
    return main_path

def plot_training_curve(metrics_history, cfg):
    try:
        aia_wavelength = cfg.AIA_WAVELENGTH
        wave_str = str(aia_wavelength)
        period_suffix = ''
        if '2024' in wave_str:
            period_suffix = '_quiet_2024'
        elif '2021' in wave_str:
            period_suffix = '_flare_2021'
        wave_num = ''.join([c for c in wave_str if c.isdigit()])
        wave_num = int(wave_num) if wave_num else 171
    except:
        wave_num = 171
        period_suffix = ''
    
    curve_dir = os.path.join(cfg.VIS_OUT_DIR, f"training_curves_{wave_num}A{period_suffix}")
    os.makedirs(curve_dir, exist_ok=True)
    
    core_curves = [
        ("psf_match_l1", "LR Reprojection L1 Loss (PSF Match)", "Loss Value", "crimson"),
        ("flux_error", "Flux Error", "Error (%)", "brown"),
        ("pcc", "Pearson Correlation Coefficient (PCC)", "PCC Value", "royalblue"),
        ("psnr", "PSNR", "PSNR (dB)", "blue")
    ]
    
    if len(metrics_history) == 0:
        print(f"⚠️ No training metrics data, skipping curve plotting")
        return
    
    epochs = [m['epoch'] for m in metrics_history]
    
    for metric_key, title, y_label, color in core_curves:
        metric_vals = [round(m['metrics'].get(metric_key, m['metrics'].get('lr_reproj_L1', 0)), 6) for m in metrics_history]
        
        plt.figure(figsize=(10, 4), dpi=300)
        plt.plot(epochs, metric_vals, linewidth=2, color=color, marker='o', markersize=2)
        plt.title(f"{title} | AIA {wave_num}Å {period_suffix.strip('_')}", fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(curve_dir, f"curve_{metric_key}_{wave_num}A{period_suffix}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
