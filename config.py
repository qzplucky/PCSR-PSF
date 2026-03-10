#!/usr/bin/env python3
"""
config.py - EDSR 2×超分多波段统一配置文件
适配AIA全波段（94/131/171/193/211/304/335Å）| 每个波段对应专属PSF | 路径完全隔离
核心：波段专属PSF自动加载+合法性校验+物理参数严格匹配
适配：移除CBAM/梯度/通量/频谱损失后的纯LR重投影L1训练逻辑
"""
import os
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm

# ===================== 核心：AIA多波段参数映射（保留波段专属PSF路径，无修改） =====================
# 各波段核心参数字典（key=波段Å，value=参数字典）
AIA_BAND_PARAMS = {
    94: {
        "HR_DIR_NAME": "0094-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2943,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0094.fits"
    },
    131: {
        "HR_DIR_NAME": "0131-HR",  
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2736,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0131.fits"
    },
    171: {
        "HR_DIR_NAME": "0171-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 200000,
        "REAL_HR_NORM_MEAN": 0.2768,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0171.fits"
    },
    193: {
        "HR_DIR_NAME": "0193-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2748,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0193.fits"
    },
    211: {
        "HR_DIR_NAME": "0211-HR", 
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2720,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0211.fits"
    },
    304: {
        "HR_DIR_NAME": "0304-HR",  
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2614,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0304.fits"
    },
    335: {
        "HR_DIR_NAME": "0335-HR",
        "GLOBAL_MIN": 0.0,
        "GLOBAL_MAX": 20000,
        "REAL_HR_NORM_MEAN": 0.2933,
        "HR_PIXEL_SCALE": 2.4,
        "PSF_RAW_SCALE": 0.6,
        "VALID_SIGNAL_THRESH": 0.0045,
        "PSF_FILE": "/home/zth24029/project/EDSR/PSF/psf_aia_0335.fits"
    }
}

# 验证支持的波段
SUPPORTED_WAVELENGTHS = list(AIA_BAND_PARAMS.keys())

# ===================== 核心：多波段EDSR 2×超分退化配置（无修改，保留波段专属PSF核心） =====================
class DegradationConfig:
    """
    多波段EDSR专用退化配置类（支持94/131/171/193/211/304/335Å）
    核心：每个波段自动加载专属PSF文件 + PSF路径合法性强制校验 + 波段专属路径隔离
    """
    def __init__(self, wavelength=304):
        # 1. 验证波段合法性
        if wavelength not in SUPPORTED_WAVELENGTHS:
            raise ValueError(f"不支持的波段{wavelength}Å！支持的波段：{SUPPORTED_WAVELENGTHS}")
        self.AIA_WAVELENGTH = wavelength
        self.band_params = AIA_BAND_PARAMS[wavelength]
        
        # --- 固定基础路径（所有波段共享）---
        self.BASE_HR_FITS_DIR = "/home/zth24029/project/data/classified_aia_fits/"
        self.BASE_LR_SAVE_DIR = "/home/zth24029/project/EDSR/LR_2x"
        self.BASE_PAIR_CSV_DIR = "/home/zth24029/project/EDSR/csv"
        
        # --- 2. 波段专属路径（自动按波段创建子目录，隔离不同波段数据）---
        self.HR_FITS_DIR = os.path.join(self.BASE_HR_FITS_DIR, self.band_params["HR_DIR_NAME"])
        self.LR_SAVE_DIR = os.path.join(self.BASE_LR_SAVE_DIR, str(wavelength))
        self.PAIR_CSV_PATH = os.path.join(self.BASE_PAIR_CSV_DIR, f"hr_lr_pairs_clean_2x_{wavelength}Å.csv")
        
        # 数据集拆分路径（波段专属）
        self.TRAIN_LR_DIR = os.path.join(self.LR_SAVE_DIR, "train")
        self.VAL_LR_DIR = os.path.join(self.LR_SAVE_DIR, "val")
        self.TRAIN_PAIR_CSV = os.path.join(self.LR_SAVE_DIR, "edsr_patches", f"edsr_train_pairs_2x_{wavelength}Å.csv")
        self.VAL_PAIR_CSV = os.path.join(self.LR_SAVE_DIR, "edsr_patches", f"edsr_val_pairs_2x_{wavelength}Å.csv")
        
        # --- 3. 加载波段专属核心参数（重点：加载专属PSF路径）---
        self.GLOBAL_MIN = self.band_params["GLOBAL_MIN"]
        self.GLOBAL_MAX = self.band_params["GLOBAL_MAX"]
        self.REAL_HR_NORM_MEAN = self.band_params["REAL_HR_NORM_MEAN"]
        self.HR_PIXEL_SCALE = self.band_params["HR_PIXEL_SCALE"]
        self.PSF_RAW_SCALE = self.band_params["PSF_RAW_SCALE"]
        self.VALID_SIGNAL_THRESH = self.band_params["VALID_SIGNAL_THRESH"]
        self.PSF_PATH = self.band_params["PSF_FILE"]  # 核心：波段专属PSF路径
        
        # --- 4. 通用固定参数（所有波段共享，与训练代码匹配）---
        self.REAL_HR_NORM_MAX = 1.0        # 归一化上限固定为1.0
        self.WEAK_THRESHOLD = self.VALID_SIGNAL_THRESH
        self.SIGMA_CLIP_SIGMA = 5          # PSF处理sigma裁剪
        self.SIGMA_CLIP_MAXITERS = 1       # sigma裁剪迭代次数
        self.VISUALIZE_SAMPLE = True       # 是否可视化LR样本
        self.SAMPLE_INDEX = 5              # 可视化样本索引
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.VAL_RATIO = 0.2               # 验证集比例
        self.SPLIT_SEED = 42               # 数据集拆分随机种子
        self.SCALE = 2                     # 超分缩放因子（固定2×）
        self.PSF_ENERGY_CROP = 0.825       # PSF能量保留阈值
        self.PSF_MAX_SIZE = 101            # 裁剪后PSF的最大尺寸
        
        # --- 数据单位协议（所有波段共享）---
        self.DATA_UNIT_PROTOCOL = {
            "hr_fits": "dn",
            "lr_fits": "normalized",
            "training_input": "normalized"
        }
        
        # 初始化时强制校验PSF路径 + 创建目录
        self._check_psf_exists()
        self.ensure_dirs()

    def _check_psf_exists(self):
        """强制校验当前波段的专属PSF文件是否存在，不存在则直接报错"""
        if not os.path.exists(self.PSF_PATH):
            raise FileNotFoundError(
                f"❌ AIA {self.AIA_WAVELENGTH}Å 专属PSF文件不存在！\n"
                f"   配置的PSF路径: {self.PSF_PATH}\n"
                f"   请检查该波段的PSF文件路径是否正确配置在 AIA_BAND_PARAMS 中"
            )
        else:
            print(f"✅ AIA {self.AIA_WAVELENGTH}Å 专属PSF文件加载成功: {self.PSF_PATH}")

    def ensure_dirs(self):  
        """确保所有波段专属目录存在"""
        for d in [self.LR_SAVE_DIR, self.TRAIN_LR_DIR, self.VAL_LR_DIR, 
                  os.path.dirname(self.TRAIN_PAIR_CSV), os.path.dirname(self.VAL_PAIR_CSV)]:
            os.makedirs(d, exist_ok=True)
    
    def read_hr_fits_safe(self, fits_path):
        """安全读取HR FITS文件，按当前波段的GLOBAL_MAX裁剪极端值"""
        try:
            with fits.open(fits_path) as hdul:
                data_dn = None
                # 优先读取HDU1，再试HDU0（兼容AIA FITS格式）
                for hdu_idx in [1, 0]:
                    if hdu_idx >= len(hdul):
                        continue
                    hdu = hdul[hdu_idx]
                    if hdu.data is None or hdu.data.ndim != 2:
                        continue
                    data_dn = hdu.data.astype(np.float32)
                    print(f"ℹ️  {os.path.basename(fits_path)} (AIA {self.AIA_WAVELENGTH}Å): 从HDU{hdu_idx}读取DN值")
                    break
                
                if data_dn is None:
                    raise ValueError(f"未找到有效2D数据: {fits_path}")
                
                # 按当前波段的GLOBAL_MAX裁剪极端值
                original_max = data_dn.max()
                if original_max > self.GLOBAL_MAX:
                    print(f"⚠️  裁剪HR极端值（AIA {self.AIA_WAVELENGTH}Å）：{os.path.basename(fits_path)} | 原max={original_max:.2f} → 约束到{self.GLOBAL_MAX:.2f}")
                    data_dn = np.clip(data_dn, self.GLOBAL_MIN, self.GLOBAL_MAX)
                
                # 清理NaN/Inf
                data_dn = np.nan_to_num(data_dn, nan=0.0, posinf=self.GLOBAL_MAX, neginf=self.GLOBAL_MIN)
                return data_dn
        
        except Exception as e:
            print(f"❌ 读取HR文件失败 (AIA {self.AIA_WAVELENGTH}Å): {fits_path} → {str(e)[:50]}")
            return np.zeros((1024, 1024), dtype=np.float32)
    
    def batch_validate_hr_files(self, max_samples=20):
        """批量验证当前波段的HR文件DN值"""
        print("\n" + "="*80)
        print(f"📊 EDSR 2×超分专用 - 批量验证HR文件DN值（AIA {self.AIA_WAVELENGTH}Å）")
        print("="*80)
        
        if not os.path.exists(self.HR_FITS_DIR):
            raise FileNotFoundError(f"当前波段{self.AIA_WAVELENGTH}Å的HR目录不存在: {self.HR_FITS_DIR}")
        
        hr_files = [f for f in os.listdir(self.HR_FITS_DIR) if f.endswith(('.fits', '.fits.gz'))]
        hr_files = hr_files[:max_samples]
        
        dn_max_list = []
        dn_mean_list = []
        file_paths = []
        
        for fname in tqdm(hr_files, desc=f"验证{self.AIA_WAVELENGTH}Å HR文件"):
            fpath = os.path.join(self.HR_FITS_DIR, fname)
            data_dn = self.read_hr_fits_safe(fpath)
            
            dn_max_list.append(data_dn.max())
            dn_mean_list.append(data_dn.mean())
            file_paths.append(fpath)
        
        # 输出波段专属统计结果
        print(f"\n📈 HR文件DN值统计（AIA {self.AIA_WAVELENGTH}Å | 共{len(hr_files)}个文件）:")
        print(f"  平均最大值: {np.mean(dn_max_list):.2f} DN")
        print(f"  最小最大值: {np.min(dn_max_list):.2f} DN")
        print(f"  最大最大值: {np.max(dn_max_list):.2f} DN")
        print(f"  平均均值: {np.mean(dn_mean_list):.2f} DN")
        print(f"  99.9%分位数: {np.quantile(dn_max_list, 0.999):.2f} DN（应≈{self.GLOBAL_MAX}）")
        
        # 归一化后统计
        norm_max_list = [dn / self.GLOBAL_MAX for dn in dn_max_list]
        print(f"\n📊 归一化后统计（AIA {self.AIA_WAVELENGTH}Å | GLOBAL_MAX={self.GLOBAL_MAX}）:")
        print(f"  平均归一化最大值: {np.mean(norm_max_list):.4f}")
        print(f"  最大归一化最大值: {np.max(norm_max_list):.4f}")
        
        print("="*80)
        
        return {
            "wavelength": self.AIA_WAVELENGTH,
            "dn_max": dn_max_list,
            "dn_mean": dn_mean_list,
            "norm_max": norm_max_list,
            "file_paths": file_paths
        }
    
    def validate_data_unit(self, data, expected_unit, data_type=None):
        """轻量级单位验证与转换（按当前波段参数）"""
        is_tensor = isinstance(data, torch.Tensor)
        data_np = data.cpu().numpy() if is_tensor else np.array(data, dtype=np.float32)
        
        # 按当前波段参数过滤极端值
        data_np = np.clip(data_np, self.GLOBAL_MIN, self.GLOBAL_MAX)
        data_np = np.nan_to_num(data_np, nan=0.0, posinf=self.GLOBAL_MAX, neginf=self.GLOBAL_MIN)
        
        # 单位判定
        q99 = np.quantile(data_np, 0.99)
        mean_val = np.mean(data_np)
        actual_unit = "normalized" if (q99 <= 1.1 and mean_val <= 0.5) else "dn"
        
        # 单位转换（按当前波段归一化参数）
        if actual_unit == "dn" and expected_unit == "normalized":
            data_converted = (data_np - self.GLOBAL_MIN) / (self.GLOBAL_MAX - self.GLOBAL_MIN + 1e-8)
            data_converted = np.clip(data_converted, 0.0, 1.0)
        
        elif actual_unit == "normalized" and expected_unit == "dn":
            data_converted = data_np * (self.GLOBAL_MAX - self.GLOBAL_MIN) + self.GLOBAL_MIN
            data_converted = np.clip(data_converted, self.GLOBAL_MIN, self.GLOBAL_MAX)
        
        else:
            data_converted = data_np
        
        # 还原数据类型
        if is_tensor:
            data_converted = torch.from_numpy(data_converted).to(data.device)
            if data.dtype != torch.float32:
                data_converted = data_converted.to(data.dtype)
        
        return data_converted

    def validate_parameters(self):
        """验证当前波段的核心参数（物理一致性+PSF合法性）"""
        print("\n" + "="*60)
        print(f"🔍 EDSR 2×超分专用 - 退化参数验证（AIA {self.AIA_WAVELENGTH}Å）")
        print("="*60)
        
        # 1. 像素尺度验证
        psf_zoom_factor = self.HR_PIXEL_SCALE / self.PSF_RAW_SCALE
        lr_pixel_scale = self.HR_PIXEL_SCALE * self.SCALE
        print(f"🌍 像素尺度（AIA {self.AIA_WAVELENGTH}Å 2×下采样）:")
        print(f"   HR_PIXEL_SCALE: {self.HR_PIXEL_SCALE:.2f}\"/pixel（物理真实）")
        print(f"   PSF_RAW_SCALE: {self.PSF_RAW_SCALE:.2f}\"/pixel（适配2×超分）")
        print(f"   PSF缩放倍数: {psf_zoom_factor:.2f}x（应等于2.0，物理匹配✅）")
        print(f"   LR_PIXEL_SCALE: {lr_pixel_scale:.2f}\"/pixel（2×下采样后尺度）")
        
        # 2. 波段专属归一化参数验证
        print(f"\n📊 归一化参数（AIA {self.AIA_WAVELENGTH}Å | 基于99.9%分位数）:")
        print(f"   GLOBAL_MIN: {self.GLOBAL_MIN:.2f} DN {'✅' if self.GLOBAL_MIN >= 0 else '❌'}")
        print(f"   GLOBAL_MAX: {self.GLOBAL_MAX:.2f} DN（99.9%分位数）")
        print(f"   归一化动态范围: {self.GLOBAL_MAX - self.GLOBAL_MIN:.2f} DN")
        print(f"   典型归一化均值: {self.REAL_HR_NORM_MEAN:.4f}（统计值）")
        
        # 3. PSF参数验证（重点：打印波段专属PSF路径）
        print(f"\n🔍 PSF退化参数（AIA {self.AIA_WAVELENGTH}Å | 专属PSF）:")
        print(f"   专属PSF路径: {self.PSF_PATH}")
        print(f"   PSF_ENERGY_CROP: {self.PSF_ENERGY_CROP:.3f}（保留{self.PSF_ENERGY_CROP*100:.1f}%能量）")
        print(f"   SIGMA_CLIP_SIGMA: {self.SIGMA_CLIP_SIGMA:.1f}（PSF裁剪）")
        print(f"   超分缩放因子: {self.SCALE}x（固定2×✅）")
        
        print("="*60)
        
        # 关键警告
        warnings = []
        if self.GLOBAL_MIN < 0:
            warnings.append(f"❌ 严重警告: {self.AIA_WAVELENGTH}Å的GLOBAL_MIN为负值！")
        if not np.isclose(psf_zoom_factor, 2.0, atol=1e-2):
            warnings.append(f"⚠️  警告: {self.AIA_WAVELENGTH}Å的PSF缩放倍数={psf_zoom_factor:.2f}≠2.0！")
        
        for warn in warnings:
            print(warn)
        
        # 验证通过
        if self.GLOBAL_MIN >= 0 and np.isclose(psf_zoom_factor, 2.0, atol=1e-2):
            print(f"✅ {self.AIA_WAVELENGTH}Å 所有核心参数物理合理，可生成HR-LR配对数据")
        else:
            print(f"⚠️  {self.AIA_WAVELENGTH}Å 部分参数异常，建议检查后再运行")
        
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

# ===================== 多波段数据加载配置（精简无用兼容参数，与训练代码匹配） =====================
class DataConfig:
    """多波段EDSR数据加载配置（引用DegradationConfig的波段参数，无冗余）"""
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        # 波段专属路径（核心保留）
        self.TRAIN_CSV_PATH = degradation_cfg.TRAIN_PAIR_CSV
        self.VAL_CSV_PATH = degradation_cfg.VAL_PAIR_CSV
        
        # 核心数据加载参数（无无用兼容参数）
        self.CHANNEL_MODE = "single"             # 天文数据固定单通道
        self.BATCH_SIZE = 4                      # 适配显存
        self.NUM_WORKERS = 4                     # 数据加载线程数
        self.PIN_MEMORY = True                   # 页锁内存加速
        self.HR_SIZE = 1024                      # HR原始尺寸
        self.LR_SIZE = 512                       # 2×超分对应LR尺寸
        self.EDSR_USE_ONLINE_NOISE = False       # 关闭在线加噪（与训练代码一致）
        self.DROP_LAST = True                    # 丢弃不完整批次
        self.SCALE = degradation_cfg.SCALE       # 引用波段的缩放因子
        self.IS_GEOM_AUG = False                 # 关闭几何增强（天文数据无需）
        self.CROP_PADDING = 16                   # 裁剪填充（兼容模型输入）

# ===================== 多波段PSF配置（核心保留，无修改） =====================
class PSFConfig:
    """多波段PSF配置（引用DegradationConfig的波段专属PSF路径，纯核心参数）"""
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        # PSF核心参数（波段专属，训练代码必需）
        self.PSF_PATH = degradation_cfg.PSF_PATH  # 核心：波段专属PSF路径
        self.TARGET_PIX_SCALE = degradation_cfg.HR_PIXEL_SCALE
        self.HR_PIXEL_SCALE = self.TARGET_PIX_SCALE
        self.SCALE = degradation_cfg.SCALE
        self.DOWNSCALE_FACTOR = degradation_cfg.SCALE
        self.PSF_ENERGY_CROP = degradation_cfg.PSF_ENERGY_CROP
        self.PSF_MAX_SIZE = degradation_cfg.PSF_MAX_SIZE
        self.PADDING = "same"                    # 卷积填充（保证尺寸一致）

# ===================== 多波段训练配置（核心精简：删除所有物理损失/冗余兼容参数） =====================
class TrainConfig:
    """
    多波段EDSR训练配置（按波段隔离输出路径）
    适配：纯LR重投影L1损失训练，移除所有CBAM/物理正则化/无用兼容参数
    """
    def __init__(self, degradation_cfg):
        self.degradation_cfg = degradation_cfg
        self.wavelength = degradation_cfg.AIA_WAVELENGTH
        
        # --- 波段专属输出路径（核心保留，自动隔离）---
        self.BASE_OUT_DIR = "/home/zth24029/project/EDSR/EDSR_psf_LR/edsr_checkpoints"
        self.BASE_VIS_DIR = "/home/zth24029/project/EDSR/EDSR_psf_LR/visualizations"
        self.BASE_LOG_DIR = "/home/zth24029/project/EDSR/EDSR_psf_LR/edsr_logs"
        
        self.OUT_DIR = os.path.join(self.BASE_OUT_DIR, str(self.wavelength))  # 波段专属模型目录
        self.VIS_OUT_DIR = os.path.join(self.BASE_VIS_DIR, str(self.wavelength))  # 波段专属可视化目录
        self.LOG_DIR = os.path.join(self.BASE_LOG_DIR, str(self.wavelength))    # 波段专属日志目录
        
        # --- 基础路径（核心保留）---
        self.HR_DIR = degradation_cfg.HR_FITS_DIR
        self.PSF_FITS = degradation_cfg.PSF_PATH  # 波段专属PSF路径
        self.RESUME = None                        # 断点续训路径
        
        # --- 数据配置（引用波段参数，无冗余）---
        self.SCALE = degradation_cfg.SCALE
        self.CROP_SIZE = 1024
        self.CLIP_MIN = degradation_cfg.GLOBAL_MIN  # 波段专属裁剪最小值
        self.CLIP_MAX = degradation_cfg.GLOBAL_MAX  # 波段专属裁剪最大值
        self.VAL_SPLIT = degradation_cfg.VAL_RATIO
        self.USE_PSF_IN_DATASET = False
        self.NUM_WORKERS = 0
        self.PSF_ENERGY_CROP = degradation_cfg.PSF_ENERGY_CROP
        self.PSF_MAX_SIZE = degradation_cfg.PSF_MAX_SIZE
        self.ADD_NOISE = False
        self.POISSON_SCALE = 0.03
        self.GAUSSIAN_SIGMA = 0.02

        # --- 模型配置（EDSR-Lite，无CBAM，纯核心参数）---
        self.MODEL_NAME = f"EDSR_x2_{self.wavelength}Å"
        self.N_RESBLOCKS = 16  # EDSR-Lite残差块数
        self.N_FEATS = 64      # EDSR-Lite特征通道数
        self.RES_SCALE = 1.0   # 残差缩放因子
        self.DATA_PARALLEL = False  # 多卡并行（适配GPU1训练）
        self.USE_ATTENTION = False  # 关闭CBAM注意力（永久禁用）

        # --- 训练超参（纯核心，与训练代码完全匹配）---
        self.EPOCHS = 50
        self.BATCH_SIZE = 4
        self.EVAL_BATCH_SIZE = 2
        self.LR = 5e-5         # 初始学习率
        self.MIN_LR = 1e-6     # 最小学习率
        self.WARMUP_EPOCHS = 5
        self.WEIGHT_DECAY = 5e-5
        self.GRAD_CLIP_NORM = 1.0
        self.AMP = True        # 混合精度训练（开启）
        self.SEED = 42         # 随机种子（保证可复现）
        
        # --- 学习率调度器（核心保留，余弦退火）---
        self.SCHEDULER = "cosine"
        self.STEP_SIZE = 20
        self.STEP_GAMMA = 0.5
        
        # --- 训练监控（核心保留，无冗余）---
        self.LOG_INTERVAL = 10   # 训练日志打印间隔
        self.VAL_INTERVAL = 5    # 验证间隔
        self.CKPT_INTERVAL = 10  # 检查点保存间隔
        self.DEVICE = degradation_cfg.DEVICE
        
        # --- 可视化配置（核心保留，波段专属）---
        self.VIS_INTERVAL = 5
        self.VIS_SAMPLE_IDX = 0
        self.USE_LOG_SCALE = False
        self.CMAP = "inferno"    # 天文数据专用配色
        self.SAVE_DIFF_MAP = True
        self.AIA_WAVELENGTH = self.wavelength  # 传入当前波段号，方便可视化
        
        # 自动创建波段专属输出目录
        os.makedirs(self.OUT_DIR, exist_ok=True)
        os.makedirs(self.VIS_OUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

# ===================== 多波段推理配置（适配训练代码的模型保存格式，无冗余） =====================
class InferenceConfig:
    """
    多波段EDSR推理配置（按波段隔离输出）
    适配：移除物理损失后的训练模型，匹配最优模型保存文件名
    """
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg
        self.wavelength = train_cfg.wavelength
        
        # 模型路径（匹配训练代码的最优模型保存格式，波段专属）
        self.CHECKPOINT_PATH = os.path.join(train_cfg.OUT_DIR)  # 指向波段专属模型目录，按需选择best_epoch文件
        
        # 输入输出路径（波段专属，自动隔离）
        self.LR_FITS_PATH = os.path.join(train_cfg.degradation_cfg.LR_SAVE_DIR, "val")
        self.OUTPUT_HR_DIR = os.path.join("edsr_hr_results", str(self.wavelength))
        self.OUTPUT_HR_FITS = f"edsr_generated_hr_{self.wavelength}Å.fits"
        self.OUTPUT_VIS_PNG = f"edsr_hr_visual_{self.wavelength}Å.png"
        
        # 推理核心参数（无冗余）
        self.PAD_MODE = "reflect"  # 反射填充（避免边缘失真）
        self.CHUNK_SIZE = 512      # 分块推理（适配显存）
        self.SAVE_DN_SPACE = True  # 保存DN空间结果（天文数据专用）
        
        # 自动创建推理输出目录
        os.makedirs(self.OUTPUT_HR_DIR, exist_ok=True)

# ===================== 多波段配置实例化（无修改，与训练代码调用完全匹配） =====================
def init_multiband_config(wavelength=304):
    """初始化指定波段的所有配置（自动加载专属PSF，无冗余参数）"""
    # 1. 初始化核心退化配置（自动校验PSF路径）
    dc = DegradationConfig(wavelength=wavelength)
    param_dict = dc.validate_parameters()
    
    # 2. 初始化其他配置（均为精简后版本，与训练代码匹配）
    data_cfg = DataConfig(dc)
    psf_cfg = PSFConfig(dc)
    train_cfg = TrainConfig(dc)
    infer_cfg = InferenceConfig(train_cfg)
    
    # 打印配置摘要（精简，与训练代码一致）
    print(f"\n✅ 多波段配置初始化完成（AIA {wavelength}Å）")
    print(f"   📁 HR目录: {dc.HR_FITS_DIR}")
    print(f"   📁 LR目录: {dc.LR_SAVE_DIR}")
    print(f"   📁 模型输出: {train_cfg.OUT_DIR}")
    print(f"   📊 归一化范围: [{dc.GLOBAL_MIN}, {dc.GLOBAL_MAX}] DN")
    print(f"   🔭 专属PSF路径: {dc.PSF_PATH}")
    print(f"   🧠 训练损失: 纯LR重投影L1 | 评估指标: PSNR/SSIM")
    
    return {
        "degradation": dc,
        "data": data_cfg,
        "psf": psf_cfg,
        "train": train_cfg,
        "inference": infer_cfg,
        "param": param_dict
    }

# ===================== 主函数：多波段配置验证（无修改，可直接运行） =====================
if __name__ == "__main__":
    # 示例1：初始化304Å配置（默认，自动加载304Å专属PSF）
    print("="*80)
    print("示例1：初始化AIA 304Å配置（加载专属PSF，精简版）")
    print("="*80)
    cfg_304 = init_multiband_config(wavelength=304)
    
    # 批量验证304Å HR文件
    stats_304 = cfg_304["degradation"].batch_validate_hr_files(max_samples=20)
    
    # 示例2：初始化193Å配置（自动加载193Å专属PSF）
    print("\n" + "="*80)
    print("示例2：初始化AIA 193Å配置（加载专属PSF，精简版）")
    print("="*80)
    cfg_193 = init_multiband_config(wavelength=193)
    
    # 验证304Å单位转换（核心功能保留）
    if stats_304["file_paths"]:
        test_hr_path = stats_304["file_paths"][0]
        test_dn = cfg_304["degradation"].read_hr_fits_safe(test_hr_path)
        test_norm = cfg_304["degradation"].validate_data_unit(test_dn, expected_unit="normalized")
        print(f"\n✅ 304Å 数据验证:")
        print(f"   DN最大值: {test_dn.max():.2f}")
        print(f"   归一化最大值: {test_norm.max():.4f}")