# EDSR-lite
#!/usr/bin/env python3
"""
edsr_psf_train.py - EDSR多波段自动轮训版本
核心升级：自动遍历所有AIA波段训练、每个波段使用专属配置、路径完全隔离
保留所有原有功能：参数量统计、训练计时、固定随机种子、可视化、GPU1专属训练
适配修改：删除梯度/通量损失函数，仅保留LR重投影L1损失，完整保留所有验证指标  a
"""

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

# ===================== 【核心修改1】导入多波段配置文件 =====================
from config import (
    init_multiband_config, SUPPORTED_WAVELENGTHS,DegradationConfig as DC  # 导入多波段配置初始化函数和支持的波段列表
)

# 导入拆分后的模块（关键适配：移除gradient_loss、flux_loss导入，仅保留验证指标）
from dataset_loader import (
    HRtoLRDataset, load_psf_fits, crop_psf_by_energy, normalize_psf_kernel,
    psf_kernel_to_conv_weights, synthesize_lr_from_hr
)
from loss_functions import (
    psnr_torch, ssim_fallback, calculate_flux_error, calculate_spectral_ratio,
)
from visualization_utils import visualize_train_step, plot_training_curve
from edsr_models import EDSR as EDSR_Lite  # 直接导入实际存在的EDSR_Lite类，无需别名

# ===================== 保留原有工具函数（无修改） =====================
def count_model_params(model, verbose=True):
    """计算PyTorch模型的参数量"""
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
        print(f"\n========== 模型参数量统计 ==========")
        print(f"总参数量: {format_params(total_params)} ({total_params:,})")
        print(f"可训练参数量: {format_params(trainable_params)} ({trainable_params:,})")
        print(f"参数量占比: {trainable_params/total_params*100:.2f}%")
        print(f"====================================\n")
    return total_params, trainable_params

def format_time(seconds):
    """格式化时间为 时:分:秒"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:.2f}"
    else:
        return f"{minutes:02d}:{secs:.2f}"

def worker_init_fn(worker_id):
    """固定DataLoader多线程随机种子"""
    seed = train_cfg.SEED + worker_id  # train_cfg是当前波段的配置
    np.random.seed(seed)
    random.seed(seed)

def numpy_json_serializer(obj):
    """处理numpy类型JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"无法序列化的类型: {type(obj)}")

# ===================== 验证函数（无修改，保留所有验证指标：含通量/频谱比） =====================
def validate(model, val_loader, scale, device, train_cfg, psf_kernel_tensor, global_min, global_max):
    """验证模型（适配多波段配置）- 适配波段专属的硬裁剪阈值，保留所有验证指标"""
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
            
            # 模型推理
            sr = model(lr)
            
            # 关键：执行和训练时完全一致的硬裁剪（用波段专属的统计阈值）
            sr = sr.clip(min=global_min, max=global_max)
            
            # 使用当前波段的静态PSF生成LR
            gen_lr = synthesize_lr_from_hr(sr, psf_kernel_tensor, scale, device)
            
            # 保存可视化样本
            if i == train_cfg.VIS_SAMPLE_IDX and vis_lr is None:
                vis_lr = lr[0].detach().cpu()
                vis_hr = hr[0].detach().cpu()
                vis_sr = sr[0].detach().cpu()
            
            # 累计LR重投影损失
            metrics["lr_reproj_L1"] += F.l1_loss(gen_lr, lr).item()
            
            # 累计传统图像指标
            metrics["psnr"] += psnr_torch(sr, hr)
            try:
                from metrics import torch_ssim, log_normalize
                metrics["ssim"] += torch_ssim(log_normalize(hr), log_normalize(sr)).mean().item()
            except:
                metrics["ssim"] += ssim_fallback(sr, hr, data_range=float(hr.max()-hr.min()))

            # 计算通量误差（保留验证指标，无修改）
            flux_err = calculate_flux_error(sr, hr)
            metrics["flux_error"] += flux_err
            
            metrics["spectral_ratio"] += calculate_spectral_ratio(sr, hr)
            metrics["count"] += 1

    # 计算平均指标
    if metrics["count"] == 0:
        return {}, None, None, None
    avg_metrics = {k: v/metrics["count"] for k, v in metrics.items() if k != "count"}
    
    # 打印验证集通量汇总（保留，验证指标）
    print(f"\n📊 {train_cfg.WAVELENGTH}Å 验证集通量汇总：")
    print(f"   总样本数: {metrics['count']} | 平均通量误差: {avg_metrics['flux_error']:.2f}%")
    
    return avg_metrics, vis_lr, vis_hr, vis_sr

# ===================== 训练函数（核心适配：删除梯度/通量损失，仅保留LR重投影L1） =====================
def train_single_band(dc, data_cfg, psf_cfg, train_cfg, infer_cfg):
    """
    训练单个波段的EDSR模型（适配修改：仅保留LR重投影L1损失，保留所有验证指标）
    :param dc: DegradationConfig实例（当前波段）
    :param data_cfg: DataConfig实例（当前波段）
    :param psf_cfg: PSFConfig实例（当前波段）
    :param train_cfg: TrainConfig实例（当前波段）
    :param infer_cfg: InferenceConfig实例（当前波段）
    """
    # 【GPU1修改】显式指定使用第二块GPU（cuda:1）
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
        torch.cuda.set_device(1)
        print(f"\n✅ 已切换到GPU1（cuda:1）进行训练")
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        device = torch.device("cuda:0")
        print(f"\n⚠️  只有GPU0可用，自动回退到GPU0训练")
    else:
        device = torch.device("cpu")
        print(f"\n⚠️  无可用GPU，使用CPU训练（速度极慢）")

    # 固定当前波段的随机种子（论文级可复现）
    seed = train_cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # 打印当前波段训练配置（适配修改：仅显示LR重投影损失）
    print(f"\n" + "="*80)
    print(f"🚀 开始训练 AIA {dc.AIA_WAVELENGTH}Å 波段（专属配置）")
    print(f"="*80)
    print(f"设备: {device}")
    print(f"当前使用GPU编号: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
    print(f"随机种子: {seed} (已全局固定，实验可完美复现)")
    print(f"HR目录: {dc.HR_FITS_DIR}")
    print(f"专属PSF路径: {dc.PSF_PATH}")
    print(f"PSF能量保留比例: {dc.PSF_ENERGY_CROP*100:.1f}%")
    print(f"超分尺度: {dc.SCALE}×")
    print(f"归一化范围: [{dc.GLOBAL_MIN}, {dc.GLOBAL_MAX}] DN")
    print(f"训练轮数: {train_cfg.EPOCHS}")
    print(f"批次大小: {train_cfg.BATCH_SIZE}")
    print(f"模型输出目录: {train_cfg.OUT_DIR}")
    print(f"可视化保存目录: {train_cfg.VIS_OUT_DIR}")
    print(f"="*80)

    # 加载当前波段的专属PSF核
    print(f"\n正在读取 {dc.AIA_WAVELENGTH}Å 专属PSF FITS: {dc.PSF_PATH}")
    psf_original = load_psf_fits(dc.PSF_PATH)
    print(f"{dc.AIA_WAVELENGTH}Å 原始PSF尺寸: {psf_original.shape[0]}×{psf_original.shape[1]}")
    
    # 能量归一化后裁剪（当前波段参数）
    psf_original = normalize_psf_kernel(psf_original)
    psf_cropped = crop_psf_by_energy(psf_original, energy_ratio=dc.PSF_ENERGY_CROP, max_size=dc.PSF_MAX_SIZE)
    psf_kernel = normalize_psf_kernel(psf_cropped)
    print(f"{dc.AIA_WAVELENGTH}Å 最终裁剪后PSF: 尺寸={psf_kernel.shape}，能量和={psf_kernel.sum():.4f}")

    # 转为卷积权重（当前波段PSF）
    psf_kernel_tensor = psf_kernel_to_conv_weights(psf_kernel, channels=1, device=device)

    # 加载当前波段的HR文件（路径隔离）
    hr_paths = sorted([Path(dc.HR_FITS_DIR) / p for p in os.listdir(dc.HR_FITS_DIR) 
                      if p[0] != '.' and p.endswith(('.fits', '.fit'))])
    if len(hr_paths) == 0:
        raise RuntimeError(f"在 {dc.HR_FITS_DIR} 中未找到 {dc.AIA_WAVELENGTH}Å 的FITS文件！")
    
    # 拆分当前波段的训练/验证集
    val_split = int(len(hr_paths) * dc.VAL_RATIO)
    train_paths = hr_paths[:-val_split] if val_split > 0 else hr_paths
    val_paths = hr_paths[-val_split:] if val_split > 0 else hr_paths
    
    # 构建当前波段的数据集（使用波段专属配置）
    train_dataset = HRtoLRDataset(train_cfg, train_paths, psf_kernel if train_cfg.USE_PSF_IN_DATASET else None)
    val_dataset = HRtoLRDataset(train_cfg, val_paths, psf_kernel if train_cfg.USE_PSF_IN_DATASET else None)
    
    # 构建DataLoader（固定当前波段种子）
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True,
                              num_workers=train_cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn,
                              generator=torch.Generator().manual_seed(train_cfg.SEED))
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 初始化当前波段的EDSR模型（参数独立）
    model = EDSR_Lite(n_resblocks=train_cfg.N_RESBLOCKS, n_feats=train_cfg.N_FEATS, scale=dc.SCALE)
    model = model.to(device)
    
    # 【GPU1修改】DataParallel指定使用GPU1
    if train_cfg.DATA_PARALLEL and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[1])
        print(f"{dc.AIA_WAVELENGTH}Å 波段使用 GPU1 进行DataParallel训练")
    elif train_cfg.DATA_PARALLEL:
        print(f"\n⚠️ {dc.AIA_WAVELENGTH}Å 波段：启用了DATA_PARALLEL但只有1块GPU可用，自动禁用多卡并行")

    # 统计当前波段模型参数量
    count_model_params(model)

    # 优化器（当前波段参数）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.LR, 
        weight_decay=train_cfg.WEIGHT_DECAY
    )
    total_steps = train_cfg.EPOCHS * len(train_loader)

    # 学习率调度器（当前波段参数）
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

    # 断点续训（当前波段模型路径，适配修改：删除权重管理器状态加载）
    start_epoch = 0
    if train_cfg.RESUME and os.path.exists(train_cfg.RESUME):
        ck = torch.load(train_cfg.RESUME, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck.get('epoch', 0) + 1
        print(f"{dc.AIA_WAVELENGTH}Å 从 {train_cfg.RESUME} 恢复训练，起始epoch: {start_epoch}")

    # 混合精度训练（当前波段配置）
    if train_cfg.AMP and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.AMP)
    else:
        scaler = torch.amp.GradScaler(enabled=False)

    # 训练准备（当前波段输出目录已自动创建）
    best_val_l1 = float('inf')
    metrics_history = []
    total_train_time = 0.0  # 当前波段总训练时间

    # 训练循环（核心适配：仅保留LR重投影损失计算，删除梯度/通量相关）
    for epoch in range(start_epoch, train_cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        running_loss_lr = 0.0  # 仅保留LR重投影损失累计

        # 记录当前Epoch开始时间
        epoch_start_time = time.time()

        for i, (lr_t, hr_t, _) in enumerate(train_loader):
            if lr_t.ndim == 3:
                lr_t = lr_t.unsqueeze(1)
            if hr_t.ndim == 3:
                hr_t = hr_t.unsqueeze(1)
            lr_t = lr_t.to(device)
            hr_t = hr_t.to(device)
            optimizer.zero_grad()

            # 前向传播（混合精度）
            with torch.amp.autocast('cuda', enabled=train_cfg.AMP):
                sr_pred = model(lr_t)
                # 使用当前波段的归一化范围裁剪输出
                sr_pred = sr_pred.clip(min=dc.GLOBAL_MIN, max=dc.GLOBAL_MAX)
                
                # 静态PSF生成LR（当前波段PSF）
                gen_lr = synthesize_lr_from_hr(sr_pred, psf_kernel_tensor, dc.SCALE, device)
                
                # 核心适配：仅计算LR重投影L1损失（唯一损失）
                loss_lr = F.l1_loss(gen_lr, lr_t)
                total_loss = loss_lr  # 总损失即为LR重投影损失

            # 反向传播
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()
            scaler.update()

            # 日志累计（仅保留总损失和LR重投影损失）
            running_loss += total_loss.item()
            running_loss_lr += loss_lr.item()

            # 日志打印（适配修改：仅显示LR重投影损失和学习率）
            if (i + 1) % train_cfg.LOG_INTERVAL == 0:
                avg_loss = running_loss / train_cfg.LOG_INTERVAL
                avg_loss_lr = running_loss_lr / train_cfg.LOG_INTERVAL
                current_lr = optimizer.param_groups[0]['lr']  # 当前学习率
                
                print(f"[{dc.AIA_WAVELENGTH}Å] [Epoch {epoch+1}/{train_cfg.EPOCHS}] Step {i+1}/{len(train_loader)} "
                      f"LR: {current_lr:.2e} | Total Loss: {avg_loss:.6f} | "
                      f"LR Reproj L1: {avg_loss_lr:.6f} )")
                
                # 重置累计损失
                running_loss = 0.0
                running_loss_lr = 0.0

        # 计算当前Epoch耗时
        epoch_train_time = time.time() - epoch_start_time
        total_train_time += epoch_train_time
        print(f"\n✅ [{dc.AIA_WAVELENGTH}Å] Epoch {epoch+1} 训练完成 | 耗时: {format_time(epoch_train_time)}")

        # 验证与可视化（保留所有验证指标，无修改）
        if (epoch + 1) % train_cfg.VAL_INTERVAL == 0 or epoch == train_cfg.EPOCHS - 1:
            val_metrics, vis_lr, vis_hr, vis_sr = validate(
                model, val_loader, dc.SCALE, device, train_cfg, psf_kernel_tensor,
                global_min=dc.GLOBAL_MIN,
                global_max=dc.GLOBAL_MAX
            )
            # 打印验证指标（适配修改：删除权重相关打印）
            print(f"\n===== [{dc.AIA_WAVELENGTH}Å] 验证 Epoch {epoch+1} =====")
            print(f"📊 核心损失：LR重投影L1 = {val_metrics.get('lr_reproj_L1', 0):.6f}")
            print(f"📈 图像指标：PSNR = {val_metrics.get('psnr', 0):.3f} | SSIM = {val_metrics.get('ssim', 0):.4f}")
            print(f"🔭 物理指标：通量误差 = {val_metrics.get('flux_error', 0)*100:.2f}% | 频谱比 = {val_metrics.get('spectral_ratio', 0):.4f}")
            print(f"=====================================================\n")

            # 保存指标历史（适配修改：删除权重字段）
            metrics_history.append({
                'epoch': epoch+1,
                'metrics': val_metrics,
                'wavelength': dc.AIA_WAVELENGTH
            })
            
            # 生成当前波段的可视化结果（无修改）
            if vis_lr is not None and vis_hr is not None and vis_sr is not None:
                visualize_train_step(epoch+1, vis_lr, vis_hr, vis_sr, val_metrics, train_cfg)
            
            # 绘制当前波段的训练曲线（无修改）
            plot_training_curve(metrics_history, train_cfg)

            # 保存当前波段最优模型（适配修改：删除权重管理器状态保存）
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
                print(f"[{dc.AIA_WAVELENGTH}Å] 保存最优模型到: {ck_path}\n")

        # 保存当前波段检查点（适配修改：删除权重管理器状态保存）
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
            print(f"[{dc.AIA_WAVELENGTH}Å] 保存检查点到: {ck_path}\n")

    # 保存当前波段完整指标历史（无修改）
    metrics_history_path = os.path.join(train_cfg.VIS_OUT_DIR, f"full_metrics_history_{dc.AIA_WAVELENGTH}Å.json")
    with open(metrics_history_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_history, f, indent=4, default=numpy_json_serializer)
    print(f"[{dc.AIA_WAVELENGTH}Å] 完整训练指标历史已保存至: {metrics_history_path}")

    # 打印当前波段训练耗时汇总（无修改）
    print(f"\n========== [{dc.AIA_WAVELENGTH}Å] 训练耗时汇总 ==========")
    print(f"总训练轮数: {train_cfg.EPOCHS - start_epoch} Epoch")
    print(f"总训练耗时: {format_time(total_train_time)}")
    print(f"平均单Epoch耗时: {format_time(total_train_time/(train_cfg.EPOCHS - start_epoch))}")
    print(f"========================================================")

    print(f"\n🎉 [{dc.AIA_WAVELENGTH}Å] 波段训练完成！")
    print(f"   最优模型路径: {train_cfg.OUT_DIR}")
    print(f"   可视化结果路径: {train_cfg.VIS_OUT_DIR}")

    # 【GPU1修改】每个波段训练完成后清理GPU1缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 [{dc.AIA_WAVELENGTH}Å] 已清理GPU1缓存")

# ===================== 多波段自动轮训主函数（无任何修改） =====================
def train_multiband(target_wavelengths=None):
    """
    多波段自动轮训入口（无修改）
    :param target_wavelengths: 指定训练的波段列表（None则训练所有支持的波段）
    """
    # 【GPU1修改】先检查GPU状态
    if torch.cuda.is_available():
        print(f"\n📋 GPU状态检查：")
        print(f"   可用GPU数量: {torch.cuda.device_count()}")
        print(f"   GPU1名称: {torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else 'N/A'}")
    else:
        print(f"\n⚠️  未检测到可用GPU，将使用CPU训练")

    # 确定要训练的波段列表
    train_wavelengths = target_wavelengths if target_wavelengths else SUPPORTED_WAVELENGTHS
    
    print(f"\n" + "="*100)
    print(f"📡 启动EDSR多波段自动轮训 | 待训练波段: {train_wavelengths} Å")
    print(f"   训练设备: GPU1 (cuda:1)")
    print(f"="*100)

    # 遍历每个波段依次训练
    for idx, wavelength in enumerate(train_wavelengths):
        try:
            # 1. 初始化当前波段的专属配置
            band_configs = init_multiband_config(wavelength=wavelength)
            
            # 2. 提取当前波段的各类配置
            dc = band_configs["degradation"]
            data_cfg = band_configs["data"]
            psf_cfg = band_configs["psf"]
            train_cfg = band_configs["train"]
            infer_cfg = band_configs["inference"]
            
            # 3. 训练当前波段
            train_cfg.WAVELENGTH = dc.AIA_WAVELENGTH  # 将当前波段号存入train_cfg，方便validate打印
            train_single_band(dc, data_cfg, psf_cfg, train_cfg, infer_cfg)
            
            # 4. 波段切换提示
            if idx < len(train_wavelengths) - 1:
                next_wave = train_wavelengths[idx+1]
                print(f"\n" + "="*100)
                print(f"🔄 [{wavelength}Å] 训练完成，即将切换到 [{next_wave}Å] 波段训练...")
                print(f"="*100)
                # 清理GPU缓存（防止显存泄漏）
                torch.cuda.empty_cache()
                time.sleep(2)  # 短暂等待，避免IO冲突
        
        except Exception as e:
            # 单个波段训练失败不影响其他波段
            print(f"\n❌ [{wavelength}Å] 波段训练出错: {str(e)}")
            print(f"⚠️  跳过当前波段，继续训练下一个波段...")
            # 出错后清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    # 所有波段训练完成
    print(f"\n" + "="*100)
    print(f"🏆 所有指定波段训练完成！")
    print(f"   训练波段: {train_wavelengths} Å")
    print(f"   训练设备: GPU1 (cuda:1)")
    print(f"   所有模型均保存在各自的波段专属目录中")
    print(f"="*100)

# ===================== 运行入口（无修改） =====================
if __name__ == '__main__':
    # 方式1：训练所有支持的波段（94/131/171/193/211/304/335）
    # train_multiband()
    
    # 方式2：指定训练部分波段（例如只训练171Å）
    train_multiband(target_wavelengths=[94,171])
