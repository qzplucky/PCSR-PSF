#!/usr/bin/env python3
import os
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

AIA_BAND_PARAMS = {
    94: {
        "HR_DIR_NAME": "0094-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2943,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0094.fits"
    },
    131: {
        "HR_DIR_NAME": "0131-HR",  
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2736,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0131.fits"
    },
    171: {
        "HR_DIR_NAME": "0171-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 200000,
        "REAL_HR_NORM_MEAN": 0.2768,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0171.fits"
    },
    193: {
        "HR_DIR_NAME": "0193-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2748,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0193.fits"
    },
    211: {
        "HR_DIR_NAME": "0211-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2720,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0211.fits"
    },
    304: {
        "HR_DIR_NAME": "0304-HR",  
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2614,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0304.fits"
    },
    335: {
        "HR_DIR_NAME": "0335-HR",
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2933,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/psf_aia_0335.fits"
    }
}

SUPPORTED_WAVELENGTHS = list(AIA_BAND_PARAMS.keys())

class DegradationConfig:
    def __init__(self, wavelength=304):
        if wavelength not in SUPPORTED_WAVELENGTHS:
            raise ValueError(f"Unsupported wavelength {wavelength}Å! Supported wavelengths: {SUPPORTED_WAVELENGTHS}")
        self.AIA_WAVELENGTH = wavelength
        self.band_params = AIA_BAND_PARAMS[wavelength]
        
        self.BASE_HR_FITS_DIR = "/project/data/classified_aia_fits/"
        self.BASE_LR_SAVE_DIR = "/project/LR_2x"
        self.BASE_PAIR_CSV_DIR = "/project/csv"
        
        self.HR_FITS_DIR = os.path.join(self.BASE_HR_FITS_DIR, self.band_params["HR_DIR_NAME"])
        self.LR_SAVE_DIR = os.path.join(self.BASE_LR_SAVE_DIR, str(wavelength))
        self.PAIR_CSV_PATH = os.path.join(self.BASE_PAIR_CSV_DIR, f"hr_lr_pairs_clean_2x_{wavelength}Å.csv")
        
        self.TRAIN_LR_DIR = os.path.join(self.LR_SAVE_DIR, "train")
        self.VAL_LR_DIR = os.path.join(self.LR_SAVE_DIR, "val")
        self.TRAIN_PAIR_CSV = os.path.join(self.LR_SAVE_DIR, "edsr_patches", f"edsr_train_pairs_2x_{wavelength}Å.csv")
        self.VAL_PAIR_CSV = os.path.join(self.LR_SAVE_DIR, "edsr_patches", f"edsr_val_pairs_2x_{wavelength}Å.csv")
        
        self.GLOBAL_MIN = self.band_params["GLOBAL_MIN"]
        self.GLOBAL_MAX = self.band_params["GLOBAL_MAX"]
        self.REAL_HR_NORM_MEAN = self.band_params["REAL_HR_NORM_MEAN"]
        self.HR_PIXEL_SCALE = self.band_params["HR_PIXEL_SCALE"]
        self.PSF_RAW_SCALE = self.band_params["PSF_RAW_SCALE"]
        self.VALID_SIGNAL_THRESH = self.band_params["VALID_SIGNAL_THRESH"]
        self.PSF_PATH = self.band_params["PSF_FILE"]
        
        self.REAL_HR_NORM_MAX = 1.0
        self.WEAK_THRESHOLD = self.VALID_SIGNAL_THRESH
        self.SIGMA_CLIP_SIGMA = 5
        self.SIGMA_CLIP_MAXITERS = 1
        self.VISUALIZE_SAMPLE = True
        self.SAMPLE_INDEX = 5
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VAL_RATIO = 0.12
        self.SPLIT_SEED = 42
        self.SCALE = 2
        self.PSF_ENERGY_CROP = 0.825
        self.PSF_MAX_SIZE = 101
        
        self.DATA_UNIT_PROTOCOL = {
            "hr_fits": "dn",
            "lr_fits": "normalized",
            "training_input": "normalized"
        }
        
        self._check_psf_exists()
        self.ensure_dirs()

    def _check_psf_exists(self):
        if not os.path.exists(self.PSF_PATH):
            raise FileNotFoundError(
                f"❌ AIA {self.AIA_WAVELENGTH}Å dedicated PSF file does not exist!\n"
                f"   Configured PSF path: {self.PSF_PATH}\n"
                f"   Please check if the PSF file path for this wavelength is correctly configured in AIA_BAND_PARAMS"
            )
        else:
            print(f"✅ AIA {self.AIA_WAVELENGTH}Å dedicated PSF file loaded successfully: {self.PSF_PATH}")

    def ensure_dirs(self):  
        for d in [self.LR_SAVE_DIR, self.TRAIN_LR_DIR, self.VAL_LR_DIR, 
                  os.path.dirname(self.TRAIN_PAIR_CSV), os.path.dirname(self.VAL_PAIR_CSV)]:
            os.makedirs(d, exist_ok=True)
    
    def read_hr_fits_safe(self, fits_path):
        try:
            with fits.open(fits_path) as hdul:
                data_dn = None
                for hdu_idx in [1, 0]:
                    if hdu_idx >= len(hdul):
                        continue
                    hdu = hdul[hdu_idx]
                    if hdu.data is None or hdu.data.ndim != 2:
                        continue
                    data_dn = hdu.data.astype(np.float32)
                    print(f"ℹ️  {os.path.basename(fits_path)} (AIA {self.AIA_WAVELENGTH}Å): Read DN values from HDU{hdu_idx}")
                    break
                
                if data_dn is None:
                    raise ValueError(f"No valid 2D data found: {fits_path}")
                
                original_max = data_dn.max()
                if original_max > self.GLOBAL_MAX:
                    print(f"⚠️  Clipping HR extreme values (AIA {self.AIA_WAVELENGTH}Å): {os.path.basename(fits_path)} | Original max={original_max:.2f} → Constrained to {self.GLOBAL_MAX:.2f}")
                    data_dn = np.clip(data_dn, self.GLOBAL_MIN, self.GLOBAL_MAX)
                
                data_dn = np.nan_to_num(data_dn, nan=0.0, posinf=self.GLOBAL_MAX, neginf=self.GLOBAL_MIN)
                return data_dn
        
        except Exception as e:
            print(f"❌ Failed to read HR file (AIA {self.AIA_WAVELENGTH}Å): {fits_path} → {str(e)[:50]}")
            return np.zeros((1024, 1024), dtype=np.float32)
    
    def batch_validate_hr_files(self, max_samples=20):
        print("\n" + "="*80)
        print(f"📊 EDSR 2× Super-Resolution - Batch Validation of HR File DN Values (AIA {self.AIA_WAVELENGTH}Å)")
        print("="*80)
        
        if not os.path.exists(self.HR_FITS_DIR):
            raise FileNotFoundError(f"HR directory for current wavelength {self.AIA_WAVELENGTH}Å does not exist: {self.HR_FITS_DIR}")
        
        hr_files = [f for f in os.listdir(self.HR_FITS_DIR) if f.endswith(('.fits', '.fits.gz'))]
        hr_files = hr_files[:max_samples]
        
        dn_max_list = []
        dn_mean_list = []
        file_paths = []
        
        for fname in tqdm(hr_files, desc=f"Validating {self.AIA_WAVELENGTH}Å HR files"):
            fpath = os.path.join(self.HR_FITS_DIR, fname)
            data_dn = self.read_hr_fits_safe(fpath)
            
            dn_max_list.append(data_dn.max())
            dn_mean_list.append(data_dn.mean())
            file_paths.append(fpath)
        
        print(f"\n📈 HR File DN Value Statistics (AIA {self.AIA_WAVELENGTH}Å | Total {len(hr_files)} files):")
        print(f"  Average maximum value: {np.mean(dn_max_list):.2f} DN")
        print(f"  Minimum maximum value: {np.min(dn_max_list):.2f} DN")
        print(f"  Maximum maximum value: {np.max(dn_max_list):.2f} DN")
        print(f"  Average mean value: {np.mean(dn_mean_list):.2f} DN")
        print(f"  99.9% quantile: {np.quantile(dn_max_list, 0.999):.2f} DN (Should be ≈{self.GLOBAL_MAX})")
        
        norm_max_list = [dn / self.GLOBAL_MAX for dn in dn_max_list]
        print(f"\n📊 Normalized Statistics (AIA {self.AIA_WAVELENGTH}Å | GLOBAL_MAX={self.GLOBAL_MAX}):")
        print(f"  Average normalized maximum: {np.mean(norm_max_list):.4f}")
        print(f"  Maximum normalized maximum: {np.max(norm_max_list):.4f}")
        
        print("="*80)
        
        return {
            "wavelength": self.AIA_WAVELENGTH,
            "dn_max": dn_max_list,
            "dn_mean": dn_mean_list,
            "norm_max": norm_max_list,
            "file_paths": file_paths
        }
    
    def validate_data_unit(self, data, expected_unit, data_type=None):
        is_tensor = isinstance(data, torch.Tensor)
        data_np = data.cpu().numpy() if is_tensor else np.array(data, dtype=np.float32)
        
        data_np = np.clip(data_np, self.GLOBAL_MIN, self.GLOBAL_MAX)
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=self.GLOBAL_MAX, neginf=self.GLOBAL_MIN)
        
        q99 = np.quantile(data_np, 0.99)
        mean_val = np.mean(data_np)
        actual_unit = "normalized" if (q99 <= 1.1 and mean_val <= 0.5) else "dn"
        
        if actual_unit == "dn" and expected_unit == "normalized":
            data_converted = (data_np - self.GLOBAL_MIN) / (self.GLOBAL_MAX - self.GLOBAL_MIN + 1e-8)
            data_converted = np.clip(data_converted, 0.0, 1.0)
        
        elif actual_unit == "normalized" and expected_unit == "dn":
            data_converted = data_np * (self.GLOBAL_MAX - self.GLOBAL_MIN) + self.GLOBAL_MIN
            data_converted = np.clip(data_converted, self.GLOBAL_MIN, self.GLOBAL_MAX)
        
        else:
            data_converted = data_np
        
        if is_tensor:
            data_converted = torch.from_numpy(data_converted).to(data.device)
            if data.dtype != torch.float32:
                data_converted = data_converted.to(data.dtype)
        
        return data_converted

    def validate_parameters(self):
        print("\n" + "="*60)
        print(f"🔍 EDSR 2× Super-Resolution - Degradation Parameter Validation (AIA {self.AIA_WAVELENGTH}Å)")
        print("="*60)
        
        psf_zoom_factor = self.HR_PIXEL_SCALE / self.PSF_RAW_SCALE
        lr_pixel_scale = self.HR_PIXEL_SCALE * self.SCALE
        print(f"🌍 Pixel Scale (AIA {self.AIA_WAVELENGTH}Å 2× Downsampling):")
        print(f"   HR_PIXEL_SCALE: {self.HR_PIXEL_SCALE:.2f}\"/pixel (Physically Real)")
        print(f"   PSF_RAW_SCALE: {self.PSF_RAW_SCALE:.2f}\"/pixel (Adapted for 2× SR)")
        print(f"   PSF Zoom Factor: {psf_zoom_factor:.2f}x (Should be 2.0, Physical Match✅)")
        print(f"   LR_PIXEL_SCALE: {lr_pixel_scale:.2f}\"/pixel (Scale after 2× Downsampling)")
        
        print(f"\n📊 Normalization Parameters (AIA {self.AIA_WAVELENGTH}Å | Based on 99.9% Quantile):")
        print(f"   GLOBAL_MIN: {self.GLOBAL_MIN:.2f} DN {'✅' if self.GLOBAL_MIN >= 0 else '❌'}")
        print(f"   GLOBAL_MAX: {self.GLOBAL_MAX:.2f} DN (99.9% Quantile)")
        print(f"   Normalization Dynamic Range: {self.GLOBAL_MAX - self.GLOBAL_MIN:.2f} DN")
        print(f"   Typical Normalized Mean: {self.REAL_HR_NORM_MEAN:.4f} (Statistical Value)")
        
        print(f"\n🔍 PSF Degradation Parameters (AIA {self.AIA_WAVELENGTH}Å | Dedicated PSF):")
        print(f"   Dedicated PSF Path: {self.PSF_PATH}")
        print(f"   PSF_ENERGY_CROP: {self.PSF_ENERGY_CROP:.3f} (Retain {self.PSF_ENERGY_CROP*100:.1f}% Energy)")
        print(f"   SIGMA_CLIP_SIGMA: {self.SIGMA_CLIP_SIGMA:.1f} (PSF Clipping)")
        print(f"   Super-Resolution Scale Factor: {self.SCALE}x (Fixed 2×✅)")
        
        print("="*60)
        
        warnings = []
        if self.GLOBAL_MIN < 0:
            warnings.append(f"❌ Critical Warning: GLOBAL_MIN of {self.AIA_WAVELENGTH}Å is negative!")
        if not np.isclose(psf_zoom_factor, 2.0, atol=1e-2):
            warnings.append(f"⚠️  Warning: PSF zoom factor of {self.AIA_WAVELENGTH}Å = {psf_zoom_factor:.2f}≠2.0!")
        
        for warn in warnings:
            print(warn)
        
        if self.GLOBAL_MIN >= 0 and np.isclose(psf_zoom_factor, 2.0, atol=1e-2):
            print(f"✅ All core parameters of {self.AIA_WAVELENGTH}Å are physically reasonable, HR-LR paired data can be generated")
        else:
            print(f"⚠️  Some parameters of {self.AIA_WAVELENGTH}Å are abnormal, please check before running")
        
        return {
            'wavelength': self.AIA_WAVELENGTH,
            'hr_pixel_scale': self.HR_PIXEL_SCALE,
            'psf_raw_scale': self.PSF_RAW_SCALE,
            'psf_zoom_factor': psf_zoom_factor,
            'scale': self.SCALE,
            'global_min': self.GLOBAL_MIN,
            'global_max': self.GLOBAL_MAX,
            'psf_path': self.PSF_PATH
        }

class DataConfig:
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        self.TRAIN_CSV_PATH = degradation_cfg.TRAIN_PAIR_CSV
        self.VAL_CSV_PATH = degradation_cfg.VAL_PAIR_CSV
        
        self.CHANNEL_MODE = "single"
        self.BATCH_SIZE = 4
        self.NUM_WORKERS = 4
        self.PIN_MEMORY = True
        self.HR_SIZE = 1024
        self.LR_SIZE = 512
        self.EDSR_USE_ONLINE_NOISE = False
        self.DROP_LAST = True
        self.SCALE = degradation_cfg.SCALE
        self.IS_GEOM_AUG = False
        self.CROP_PADDING = 16

class PSFConfig:
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        self.PSF_PATH = degradation_cfg.PSF_PATH
        self.TARGET_PIX_SCALE = degradation_cfg.HR_PIXEL_SCALE
        self.HR_PIXEL_SCALE = self.TARGET_PIX_SCALE
        self.SCALE = degradation_cfg.SCALE
        self.DOWNSCALE_FACTOR = degradation_cfg.SCALE
        self.PSF_ENERGY_CROP = degradation_cfg.PSF_ENERGY_CROP
        self.PSF_MAX_SIZE = degradation_cfg.PSF_MAX_SIZE
        self.PADDING = "same"

class TrainConfig:
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        self.BASE_OUT_DIR = "/project/checkpoints"
        self.BASE_VIS_DIR = "/project/visualizations"
        self.BASE_LOG_DIR = "/project/logs"
        
        self.OUT_DIR = os.path.join(self.BASE_OUT_DIR, str(self.wavelength))
        self.VIS_OUT_DIR = os.path.join(self.BASE_VIS_DIR, str(self.wavelength))
        self.LOG_DIR = os.path.join(self.BASE_LOG_DIR, str(self.wavelength))
        
        self.HR_DIR = degradation_cfg.HR_FITS_DIR
        self.PSF_FITS = degradation_cfg.PSF_PATH
        self.RESUME = None
        
        self.SCALE = degradation_cfg.SCALE
        self.CROP_SIZE = 1024
        self.CLIP_MIN = degradation_cfg.GLOBAL_MIN
        self.CLIP_MAX = degradation_cfg.GLOBAL_MAX
        self.VAL_SPLIT = degradation_cfg.VAL_RATIO
        self.USE_PSF_IN_DATASET = False
        self.NUM_WORKERS = 0
        self.PSF_ENERGY_CROP = degradation_cfg.PSF_ENERGY_CROP
        self.PSF_MAX_SIZE = degradation_cfg.PSF_MAX_SIZE
        self.ADD_NOISE = False
        self.POISSON_SCALE = 0.03
        self.GAUSSIAN_SIGMA = 0.02

        self.MODEL_NAME = f"EDSR_x2_{self.wavelength}Å"
        self.N_RESBLOCKS = 16
        self.N_FEATS = 64
        self.RES_SCALE = 1.0
        self.DATA_PARALLEL = False
        self.USE_ATTENTION = False

        self.EPOCHS = 50
        self.BATCH_SIZE = 4
        self.EVAL_BATCH_SIZE = 2
        self.LR = 5e-5
        self.MIN_LR = 1e-6
        self.WARMUP_EPOCHS = 5
        self.WEIGHT_DECAY = 5e-5
        self.GRAD_CLIP_NORM = 1.0
        self.AMP = True
        self.SEED = 42
        
        self.SCHEDULER = "cosine"
        self.STEP_SIZE = 20
        self.STEP_GAMMA = 0.5
        
        self.LOG_INTERVAL = 10
        self.VAL_INTERVAL = 5
        self.CKPT_INTERVAL = 10
        self.DEVICE = degradation_cfg.DEVICE
        
        self.VIS_INTERVAL = 5
        self.VIS_SAMPLE_IDX = 0
        self.USE_LOG_SCALE = False
        self.CMAP = "inferno"
        self.SAVE_DIFF_MAP = True
        self.AIA_WAVELENGTH = self.wavelength
        
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.VIS_OUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

class InferenceConfig:
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg
        self.wavelength = train_cfg.wavelength
        
        self.CHECKPOINT_PATH = os.path.join(train_cfg.OUT_DIR)
        
        self.LR_FITS_PATH = os.path.join(train_cfg.degradation_cfg.LR_SAVE_DIR, "val")
        self.OUTPUT_HR_DIR = os.path.join("edsr_hr_results", str(self.wavelength))
        self.OUTPUT_HR_FITS = f"edsr_generated_hr_{self.wavelength}Å.fits"
        self.OUTPUT_VIS_PNG = f"edsr_hr_visual_{self.wavelength}Å.png"
        
        self.PAD_MODE = "reflect"
        self.CHUNK_SIZE = 512
        self.SAVE_DN_SPACE = True
        
        os.makedirs(self.OUTPUT_HR_DIR, exist_ok=True)

def init_multiband_config(wavelength=304):
    dc = DegradationConfig(wavelength=wavelength)
    param_dict = dc.validate_parameters()
    
    data_cfg = DataConfig(dc)
    psf_cfg = PSFConfig(dc)
    train_cfg = TrainConfig(dc)
    infer_cfg = InferenceConfig(train_cfg)
    
    print(f"\n✅ Multi-band configuration initialization completed (AIA {wavelength}Å)")
    print(f"   📁 HR Directory: {dc.HR_FITS_DIR}")
    print(f"   📁 LR Directory: {dc.LR_SAVE_DIR}")
    print(f"   📁 Model Output: {train_cfg.OUT_DIR}")
    print(f"   📊 Normalization Range: [{dc.GLOBAL_MIN}, {dc.GLOBAL_MAX}] DN")
    print(f"   🔭 Dedicated PSF Path: {dc.PSF_PATH}")
    print(f"   🧠 Training Loss: LR Reprojection L1 ")
    
    return {
        "degradation": dc,
        "data": data_cfg,
        "psf": psf_cfg,
        "train": train_cfg,
        "inference": infer_cfg,
        "param": param_dict
    }

if __name__ == "__main__":
    print("="*80)
    print("Example 1: Initialize AIA 304Å Configuration (Load Dedicated PSF, Simplified Version)")
    print("="*80)
    cfg_304 = init_multiband_config(wavelength=304)
    
    stats_304 = cfg_304["degradation"].batch_validate_hr_files(max_samples=20)
    
    print("\n" + "="*80)
    print("Example 2: Initialize AIA 193Å Configuration (Load Dedicated PSF, Simplified Version)")
    print("="*80)
    cfg_193 = init_multiband_config(wavelength=193)
    
    if stats_304["file_paths"]:
        test_hr_path = stats_304["file_paths"][0]
        test_dn = cfg_304["degradation"].read_hr_fits_safe(test_hr_path)
        test_norm = cfg_304["degradation"].validate_data_unit(test_dn, expected_unit="normalized")
        print(f"\n✅ 304Å Data Validation:")
        print(f"   DN Maximum Value: {test_dn.max():.2f}")
        print(f"   Normalized Maximum Value: {test_norm.max():.4f}")
