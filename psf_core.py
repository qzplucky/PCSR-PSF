import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from config import train_cfg, data_cfg
from config import DC

def ascii_only(text: str) -> str:
    import re
    cleaned = re.sub(r'[^ -~]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else 'None'

class PSFDegradationOperator:
    def __init__(self, psf_path=DC.PSF_PATH, target_pix_scale=DC.HR_PIXEL_SCALE, device=None):
        self.device = device if device is not None else train_cfg.DEVICE
        self.psf = self._load_and_adapt_psf(psf_path, target_pix_scale)
        self.psf_kernel = self.psf.unsqueeze(0).unsqueeze(0)
        self.downscale_factor = DC.SCALE
        
        assert self.downscale_factor == 2, f"DC.SCALE={self.downscale_factor}"
        
        self.padding = (self.psf.shape[0] // 2, self.psf.shape[1] // 2)
        
        print(f"=== EDSR专用PSF退化算子初始化 ===")
        print(f"  设备：{self.device}")
        print(f"  最终PSF核形状：{self.psf.shape}（H×W，奇数尺寸）")
        print(f"  超分缩放因子：{self.downscale_factor}x（实验固定）")
        print(f"  PSF能量保留阈值：{DC.PSF_ENERGY_CROP:.2f}")
        print(f"  HR像素尺度：{target_pix_scale:.6f} arcsec/px")
        print(f"==================================")

    def _load_and_adapt_psf(self, psf_path, target_pix_scale):
        with fits.open(psf_path) as hdul:
            psf_data = hdul[0].data.astype(np.float32)
            psf_cdelt1 = hdul[0].header.get('CDELT1')
            psf_cdelt2 = hdul[0].header.get('CDELT2')
        
        if psf_cdelt1 is None or psf_cdelt2 is None:
            raise ValueError("PSF文件缺少CDELT1/CDELT2像素尺度参数")
        
        psf_pix_x = abs(psf_cdelt1)
        psf_pix_y = abs(psf_cdelt2)
        psf_avg_pix = (psf_pix_x + psf_pix_y) / 2
        scale_factor = psf_avg_pix / target_pix_scale
        print(f"PSF尺度适配：原始{psf_avg_pix:.6f} arcsec/px → 目标{target_pix_scale:.6f} arcsec/px（缩放因子{scale_factor:.6f}）")

        psf_sum = psf_data.sum()
        hc, wc = psf_data.shape[0] // 2, psf_data.shape[1] // 2
        pre_crop = 0
        while True:
            pre_crop += 10
            if pre_crop > min(hc, wc):
                pre_crop = min(hc, wc)
                break
            crop_psf = psf_data[hc-pre_crop:hc+pre_crop, wc-pre_crop:wc+pre_crop]
            if crop_psf.sum() / psf_sum >= 0.99:
                break
        psf_data = psf_data[hc-pre_crop:hc+pre_crop, wc-pre_crop:wc+pre_crop]
        print(f"PSF预裁剪：{psf_data.shape}（保留≥99%能量）")

        psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).unsqueeze(0)
        psf_scaled = F.interpolate(
            psf_tensor,
            scale_factor=1/scale_factor,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True
        )
        psf_scaled = psf_scaled / psf_scaled.sum()
        psf_scaled_np = psf_scaled.cpu().numpy()[0, 0]

        crop = int(round(300 * psf_avg_pix / target_pix_scale))
        crop = crop if crop % 2 == 1 else crop + 1
        hc_np = psf_scaled_np.shape[0] // 2
        
        while True:
            if crop > hc_np:
                crop = hc_np // 2 * 2 + 1
                break
            core = psf_scaled_np[
                hc_np - crop//2 : hc_np + crop//2 + 1,
                hc_np - crop//2 : hc_np + crop//2 + 1
            ]
            energy_ratio = core.sum() / psf_scaled_np.sum()
            if energy_ratio >= DC.PSF_ENERGY_CROP:
                break
            crop = int(crop * 1.2)
            crop = crop if crop % 2 == 1 else crop + 1
        
        core = core / core.sum()
        if np.isnan(core).any() or np.isinf(core).any():
            raise ValueError("PSF核包含NaN/Inf值，加载失败")
        
        print(f"PSF最终裁剪：{core.shape}（奇数尺寸，能量保留{energy_ratio:.3f}≥{DC.PSF_ENERGY_CROP}）")
        return torch.from_numpy(core).to(self.device)

    def _preprocess_hr(self, x_hr):
        B, C, H, W = x_hr.shape
        
        x_hr_denorm = x_hr * (DC.GLOBAL_MAX - DC.GLOBAL_MIN) + DC.GLOBAL_MIN
        x_hr_denorm = torch.clamp(x_hr_denorm, DC.GLOBAL_MIN + 1e-8, DC.GLOBAL_MAX)
        
        x_hr_norm = (x_hr_denorm - DC.GLOBAL_MIN) / (DC.GLOBAL_MAX - DC.GLOBAL_MIN)
        x_hr_norm = torch.clamp(x_hr_norm, 1e-8, 1.0)
        
        return x_hr_norm

    @torch.no_grad()
    def __call__(self, x_hr):
        B, C, H, W = x_hr.shape
        
        x_hr = self._preprocess_hr(x_hr)
        
        psf_kernel_expand = self.psf_kernel.repeat(C, 1, 1, 1)
        x_blur = F.conv2d(
            x_hr, 
            psf_kernel_expand, 
            padding=self.padding,
            groups=C,
            padding_mode='reflect'
        )
        x_blur = torch.clamp(x_blur, 1e-8, 1.0)
        
        if H % 2 != 0 or W % 2 != 0:
            x_blur = x_blur[:, :, :H - H%2, :W - W%2]
        x_down = F.avg_pool2d(x_blur, kernel_size=2, stride=2)
        x_down = torch.clamp(x_down, 1e-8, 1.0)
        
        return x_down

    def to(self, device):
        self.device = device
        self.psf = self.psf.to(device)
        self.psf_kernel = self.psf_kernel.to(device)
        return self

class EDSRPSFAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.psf_degrader = PSFDegradationOperator(device=train_cfg.DEVICE)
        self.device = train_cfg.DEVICE
        
        self.lambda_lr = 0.7
        self.lambda_hr = 0.3
        print(f"\n=== EDSR损失模块初始化 ===")
        print(f"  损失权重：{self.lambda_lr}×LR重投影L1 + {self.lambda_hr}×HR L1（实验固定）")
        print(f"===========================")

    def forward(self, x_pred_hr, x_lr, x_gt_hr):
        x_gen_lr = self.psf_degrader(x_pred_hr)
        
        loss_lr_reproj = F.l1_loss(x_gen_lr, x_lr)
        loss_hr_fidelity = F.l1_loss(x_pred_hr, x_gt_hr)
        
        loss_total = self.lambda_lr * loss_lr_reproj + self.lambda_hr * loss_hr_fidelity
        
        psf_metrics = {
            "loss_total": loss_total.item(),
            "loss_lr_reproj": loss_lr_reproj.item(),
            "loss_hr_fidelity": loss_hr_fidelity.item(),
            "gen_lr_mean": x_gen_lr.mean().item(),
            "input_lr_mean": x_lr.mean().item(),
            "pred_hr_mean": x_pred_hr.mean().item(),
            "gt_hr_mean": x_gt_hr.mean().item()
        }
        
        return loss_total, psf_metrics

    def to(self, device):
        super().to(device)
        self.device = device
        self.psf_degrader = self.psf_degrader.to(device)
        return self
