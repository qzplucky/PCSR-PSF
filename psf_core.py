import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
from config import train_cfg, data_cfg
from config import DC  # 统一用DC配置

def ascii_only(text: str) -> str:
    """保留ASCII字符，清理特殊符号（原函数保留，未改动）"""
    import re
    cleaned = re.sub(r'[^ -~]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else 'None'

class PSFDegradationOperator:
    """
    EDSR专用PSF退化算子（严格匹配实验要求：仅PSF模糊+2×binning下采样，无噪声/额外技巧）
    核心：适配EDSR像素域超分的物理一致性约束，固定2×超分，保证PSF卷积的对称性
    """
    def __init__(self, psf_path=DC.PSF_PATH, target_pix_scale=DC.HR_PIXEL_SCALE, device=None):
        self.device = device if device is not None else train_cfg.DEVICE
        self.psf = self._load_and_adapt_psf(psf_path, target_pix_scale)
        self.psf_kernel = self.psf.unsqueeze(0).unsqueeze(0)
        self.downscale_factor = DC.SCALE
        
        # 实验强制约束：仅支持2×超分（核心实验要求）
        assert self.downscale_factor == 2, f"实验要求固定scale=2，当前DC.SCALE={self.downscale_factor}，请修改配置"
        
        # 奇数PSF核的对称padding（保证卷积后尺寸与原HR一致）
        self.padding = (self.psf.shape[0] // 2, self.psf.shape[1] // 2)
        
        # 打印初始化信息（聚焦实验关键物理参数）
        print(f"=== EDSR专用PSF退化算子初始化 ===")
        print(f"  设备：{self.device}")
        print(f"  最终PSF核形状：{self.psf.shape}（H×W，奇数尺寸）")
        print(f"  超分缩放因子：{self.downscale_factor}x（实验固定）")
        print(f"  PSF能量保留阈值：{DC.PSF_ENERGY_CROP:.2f}")
        print(f"  HR像素尺度：{target_pix_scale:.6f} arcsec/px")
        print(f"==================================")

    def _load_and_adapt_psf(self, psf_path, target_pix_scale):
        """
        加载并适配PSF核（强制奇数尺寸，保证卷积对称性，匹配实验物理要求）
        """
        with fits.open(psf_path) as hdul:
            psf_data = hdul[0].data.astype(np.float32)
            psf_cdelt1 = hdul[0].header.get('CDELT1')
            psf_cdelt2 = hdul[0].header.get('CDELT2')
        
        if psf_cdelt1 is None or psf_cdelt2 is None:
            raise ValueError("PSF文件缺少CDELT1/CDELT2像素尺度参数（天文FITS标准）")
        
        # 计算PSF像素尺度与目标HR尺度的比例
        psf_pix_x = abs(psf_cdelt1)
        psf_pix_y = abs(psf_cdelt2)
        psf_avg_pix = (psf_pix_x + psf_pix_y) / 2
        scale_factor = psf_avg_pix / target_pix_scale
        print(f"PSF尺度适配：原始{psf_avg_pix:.6f} arcsec/px → 目标{target_pix_scale:.6f} arcsec/px（缩放因子{scale_factor:.6f}）")

        # 动态预裁剪（保留99%能量，减少计算量）
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

        # 缩放PSF到目标像素尺度（双三次插值，保持能量守恒）
        psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).unsqueeze(0)
        psf_scaled = F.interpolate(
            psf_tensor,
            scale_factor=1/scale_factor,
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=True
        )
        psf_scaled = psf_scaled / psf_scaled.sum()  # 能量归一化
        psf_scaled_np = psf_scaled.cpu().numpy()[0, 0]

        # 按能量裁剪PSF核（强制奇数尺寸，保证卷积对称）
        crop = int(round(300 * psf_avg_pix / target_pix_scale))
        crop = crop if crop % 2 == 1 else crop + 1  # 强制奇数
        hc_np = psf_scaled_np.shape[0] // 2
        
        while True:
            if crop > hc_np:
                crop = hc_np // 2 * 2 + 1  # 保留奇数
                break
            # 奇数核的切片逻辑（中心对称）
            core = psf_scaled_np[
                hc_np - crop//2 : hc_np + crop//2 + 1,
                hc_np - crop//2 : hc_np + crop//2 + 1
            ]
            energy_ratio = core.sum() / psf_scaled_np.sum()
            if energy_ratio >= DC.PSF_ENERGY_CROP:
                break
            crop = int(crop * 1.2)
            crop = crop if crop % 2 == 1 else crop + 1  # 始终保持奇数
        
        # 最终能量归一化 + 数值稳定性校验
        core = core / core.sum()
        if np.isnan(core).any() or np.isinf(core).any():
            raise ValueError("PSF核包含NaN/Inf值，加载失败（请检查PSF文件或适配逻辑）")
        
        print(f"PSF最终裁剪：{core.shape}（奇数尺寸，能量保留{energy_ratio:.3f}≥{DC.PSF_ENERGY_CROP}）")
        return torch.from_numpy(core).to(self.device)

    def _preprocess_hr(self, x_hr):
        """
        EDSR专用HR预处理（极简版：仅数值保护，无额外技巧，严格匹配实验要求）
        移除Sigma裁剪/掩码/均值对齐，避免引入无关变量
        """
        B, C, H, W = x_hr.shape
        
        # 反归一化到DN空间 + 数值裁剪（防止溢出）
        x_hr_denorm = x_hr * (DC.GLOBAL_MAX - DC.GLOBAL_MIN) + DC.GLOBAL_MIN
        x_hr_denorm = torch.clamp(x_hr_denorm, DC.GLOBAL_MIN + 1e-8, DC.GLOBAL_MAX)
        
        # 重新归一化到0~1（保持与输入一致的数值范围）
        x_hr_norm = (x_hr_denorm - DC.GLOBAL_MIN) / (DC.GLOBAL_MAX - DC.GLOBAL_MIN)
        x_hr_norm = torch.clamp(x_hr_norm, 1e-8, 1.0)
        
        return x_hr_norm

    @torch.no_grad()
    def __call__(self, x_hr):
        """
        执行PSF退化（严格按实验流程：PSF卷积 + 2×binning下采样）
        Args:
            x_hr: 输入HR图像 [B, C, H, W]（归一化空间0~1）
        Returns:
            x_lr: 退化后的LR图像 [B, C, H//2, W//2]（归一化空间0~1）
        """
        B, C, H, W = x_hr.shape
        
        # 极简预处理（仅数值保护）
        x_hr = self._preprocess_hr(x_hr)
        
        # 1. PSF模糊（分组卷积，保证多通道一致性）
        psf_kernel_expand = self.psf_kernel.repeat(C, 1, 1, 1)
        x_blur = F.conv2d(
            x_hr, 
            psf_kernel_expand, 
            padding=self.padding,
            groups=C,
            padding_mode='reflect'  # 天文数据边缘反射填充，更符合物理实际
        )
        x_blur = torch.clamp(x_blur, 1e-8, 1.0)
        
        # 2. 严格2×binning下采样（实验要求的avg_pool2d，替代area插值）
        # 注意：需确保H/W是2的整数倍，否则截断最后一行/列
        if H % 2 != 0 or W % 2 != 0:
            x_blur = x_blur[:, :, :H - H%2, :W - W%2]  # 裁剪到偶数尺寸
        x_down = F.avg_pool2d(x_blur, kernel_size=2, stride=2)
        x_down = torch.clamp(x_down, 1e-8, 1.0)
        
        return x_down

    def to(self, device):
        """设备迁移（保证PSF核与模型在同一设备）"""
        self.device = device
        self.psf = self.psf.to(device)
        self.psf_kernel = self.psf_kernel.to(device)
        return self

# --------------------------- EDSR专用PSF约束损失模块 ---------------------------
class EDSRPSFAwareLoss(nn.Module):
    """
    EDSR专用PSF损失计算模块（严格匹配实验损失规则：0.7×LR重投影L1 + 0.3×HR保真L1）
    核心：物理一致性约束（LR重投影误差）+ 像素域保真（HR L1）
    """
    def __init__(self):
        super().__init__()
        self.psf_degrader = PSFDegradationOperator(device=train_cfg.DEVICE)
        self.device = train_cfg.DEVICE
        
        # 实验固定损失权重（不可修改，保证实验一致性）
        self.lambda_lr = 0.7  # LR重投影损失权重（核心物理约束）
        self.lambda_hr = 0.3  # HR保真损失权重（辅助像素约束）
        print(f"\n=== EDSR损失模块初始化 ===")
        print(f"  损失权重：{self.lambda_lr}×LR重投影L1 + {self.lambda_hr}×HR L1（实验固定）")
        print(f"===========================")

    def forward(self, x_pred_hr, x_lr, x_gt_hr):
        """
        计算实验要求的PSF约束损失
        Args:
            x_pred_hr: EDSR输出的预测HR [B, C, H, W]
            x_lr:      输入LR图像 [B, C, H//2, W//2]
            x_gt_hr:   真实HR图像 [B, C, H, W]
        Returns:
            loss_total: 综合损失（实验固定权重）
            psf_metrics: 损失分项指标字典（便于训练监控）
        """
        # 1. 干净PSF退化：预测HR → 退化LR（与数据集生成LR的流程完全一致）
        x_gen_lr = self.psf_degrader(x_pred_hr)
        
        # 2. 实验要求的核心损失项（全部使用L1，避免MSE的离群值敏感）
        loss_lr_reproj = F.l1_loss(x_gen_lr, x_lr)    # LR重投影L1损失（物理一致性）
        loss_hr_fidelity = F.l1_loss(x_pred_hr, x_gt_hr)  # HR保真L1损失（像素域约束）
        
        # 3. 总损失（严格按实验权重计算）
        loss_total = self.lambda_lr * loss_lr_reproj + self.lambda_hr * loss_hr_fidelity
        
        # 封装监控指标（便于训练日志打印核心物理量）
        psf_metrics = {
            "loss_total": loss_total.item(),
            "loss_lr_reproj": loss_lr_reproj.item(),  # 核心：LR重投影误差
            "loss_hr_fidelity": loss_hr_fidelity.item(),  # 辅助：HR保真误差
            "gen_lr_mean": x_gen_lr.mean().item(),    # 退化LR的均值（数值稳定性监控）
            "input_lr_mean": x_lr.mean().item(),      # 输入LR的均值（对比验证）
            "pred_hr_mean": x_pred_hr.mean().item(),  # 预测HR的均值（数值偏移监控）
            "gt_hr_mean": x_gt_hr.mean().item()       # 真实HR的均值（对比验证）
        }
        
        return loss_total, psf_metrics

    def to(self, device):
        """设备同步（保证损失模块与模型/数据在同一设备）"""
        super().to(device)
        self.device = device
        self.psf_degrader = self.psf_degrader.to(device)
        return self