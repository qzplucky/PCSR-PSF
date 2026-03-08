# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# 改进版EDSR，论文方法
# # ==============================================================================
# # 残差块：核心修改（数值稳定性+降低过拟合）
# # ==============================================================================
# class ResBlock(nn.Module):
#     """原版残差块：新增GroupNorm特征平滑，适配EDSR-Lite，抑制高频噪声"""
#     # 🔥 修改1：降低Dropout默认值到0.05（减少特征稀疏性，避免nan）
#     # 🔥 修改2：新增res_scale默认值0.1（降低残差叠加幅度，防止数值溢出）
#     def __init__(self, n_feats, kernel_size=3, res_scale=0.1, dropout_rate=0.05):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
#         # 🔥 修改3：GN添加eps=1e-6（防止标准差为0导致1/0，彻底杜绝nan）
#         self.gn = nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6)
#         self.act = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
#         self.res_scale = res_scale
#         self.dropout = nn.Dropout2d(p=dropout_rate)  # 保留Dropout，防止过拟合

#     def forward(self, x):
#         res = self.conv1(x)
#         res = self.gn(res)  # 特征平滑：抑制高频噪声，减少像素伪影
#         res = self.act(res)
#         res = self.dropout(res)
#         res = self.conv2(res)
#         return x + res * self.res_scale

# # ==============================================================================
# # EDSR-Lite：整合消网格+数值稳定优化
# # ==============================================================================
# class EDSR_Lite(nn.Module):
#     """EDSR-Lite超分模型：轻量版EDSR，适配单通道太阳图像，支持2/4倍及任意尺度上采样"""
#     # 🔥 修改4：同步降低Dropout默认值到0.05
#     def __init__(self, n_resblocks=8, n_feats=64, scale=2, dropout_rate=0.05):
#         super().__init__()
#         self.scale = scale
#         self.n_feats = n_feats
#         self.sub_mean = nn.Identity()
#         self.add_mean = nn.Identity()

#         # 1. 头部卷积：单通道转特征通道（无修改）
#         self.head = nn.Conv2d(1, n_feats, 3, padding=1)

#         # 2. 主体：残差块堆叠 + 全局残差连接（无修改）
#         self.body = nn.Sequential(*[
#             ResBlock(n_feats, dropout_rate=dropout_rate) 
#             for _ in range(n_resblocks)
#         ])
#         self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
#         self.body_dropout = nn.Dropout2d(p=dropout_rate)

#         # 3. 上采样模块：消网格+数值稳定优化（核心修改）
#         self.upsample = self._make_upsample_module(n_feats, scale)

#         # 4. 跳跃连接融合层（无修改）
#         self.skip_fusion = nn.Conv2d(n_feats * 2, n_feats, 3, padding=1)

#         # 5. 尾部卷积（无修改）
#         self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

#     def _make_upsample_module(self, n_feats, scale):
#         """构建上采样模块：新增GN + 删ReLU + 低强度初始化（核心修改）"""
#         m = []
#         if scale == 2:
#             # 2倍上采样：通道扩增→PixelShuffle→GN平滑→卷积校准（删ReLU）
#             conv_ps = nn.Conv2d(n_feats, n_feats * scale ** 2, 3, padding=1)
#             # 🔥 修改5：降低初始化强度（从1/n_feats→1/(n_feats*10)，适配天文数据高动态范围）
#             nn.init.constant_(conv_ps.weight, 1.0 / (n_feats * 10))
#             nn.init.zeros_(conv_ps.bias)
#             m.append(conv_ps)
            
#             m.append(nn.PixelShuffle(scale))  # 空间重排
#             # 🔥 修改6：GN添加eps=1e-6（数值稳定）
#             m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
#             # 🔥 保留：删掉ReLU，避免放大网格（关键！）
#             # m.append(nn.ReLU(inplace=True))  # 注释/删除这行
#             m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))  # 特征校准
            
#         elif scale == 4:
#             # 4倍上采样：1次PixelShuffle+GN + 1次插值（避免堆叠）
#             # 第一次2倍PixelShuffle
#             conv_ps = nn.Conv2d(n_feats, n_feats * 4, 3, padding=1)
#             # 🔥 修改7：同步降低初始化强度
#             nn.init.constant_(conv_ps.weight, 1.0 / (n_feats * 10))
#             nn.init.zeros_(conv_ps.bias)
#             m.append(conv_ps)
            
#             m.append(nn.PixelShuffle(2))
#             # 🔥 修改8：GN添加eps=1e-6
#             m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
#             # m.append(nn.ReLU(inplace=True))  # 删掉ReLU
#             m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
            
#             # 第二次2倍用插值（无网格），替代PixelShuffle堆叠
#             m.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
#             m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
#         else:
#             # 其他尺度：保持原逻辑 + GN加eps
#             m.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
#             # 🔥 修改9：GN添加eps=1e-6
#             m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
#             m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
#         return nn.Sequential(*m)

#     def forward(self, x):
#         # 预处理（无修改）
#         x = self.sub_mean(x)
        
#         # 头部特征提取
#         x_head = self.head(x)
        
#         # 残差块特征提取 + 全局残差连接
#         res = self.body(x_head)
#         res = self.body_conv(res)
#         res = self.body_dropout(res)
#         x_body = x_head + res
        
#         # 上采样（已包含GN+消网格优化）
#         x_up = self.upsample(x_body)
        
#         # 跳跃连接融合
#         x_head_up = F.interpolate(x_head, scale_factor=self.scale, mode='bilinear', align_corners=False)
#         x_fused = torch.cat([x_up, x_head_up], dim=1)
#         x_fused = self.skip_fusion(x_fused)
        
#         # 尾部卷积
#         x_out = self.tail(x_fused)
#         x_out = self.add_mean(x_out)
        
#         return x_out

# # ==============================================================================
# # 测试代码（验证模型可运行+维度正确）
# # ==============================================================================
# if __name__ == "__main__":
#     # 初始化模型：8个残差块，64维特征，2倍超分
#     model = EDSR_Lite(n_resblocks=8, n_feats=64, scale=2, dropout_rate=0.05)
#     # 构造测试输入：[batch_size, channel, height, width]
#     test_input = torch.randn(1, 1, 512, 512)
#     # 模型前向传播
#     test_output = model(test_input)
#     # 打印尺寸信息
#     print(f"EDSR-Lite - 输入尺寸：{test_input.shape} | 输出尺寸：{test_output.shape}")
#     # 计算总参数量（GN无参数量，仅增加计算量，可忽略）
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"EDSR-Lite - 总参数量：{total_params / 1000:.2f}k")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 原版残差块：无GN、无Dropout，仅保留核心残差连接
# ==============================================================================
class ResBlock(nn.Module):
    """原版EDSR残差块：无正则化，仅通过残差缩放稳定训练"""
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        return x + res * self.res_scale

# ==============================================================================
# 原版EDSR：恢复标准超分架构，无物理约束适配
# ==============================================================================
class EDSR(nn.Module):
    """原版EDSR超分模型：标准超分架构，无针对太阳图像的特殊优化"""
    def __init__(self, n_resblocks=16, n_feats=64, scale=2, res_scale=0.1):
        super().__init__()
        self.scale = scale
        self.n_feats = n_feats
        self.sub_mean = nn.Identity()
        self.add_mean = nn.Identity()

        # 1. 头部卷积：单通道转特征通道
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)

        # 2. 主体：残差块堆叠 + 全局残差连接
        self.body = nn.Sequential(*[
            ResBlock(n_feats, res_scale=res_scale) 
            for _ in range(n_resblocks)
        ])
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # 3. 上采样模块：恢复原版PixelShuffle堆叠（无GN、无消网格修改）
        self.upsample = self._make_upsample_module(n_feats, scale)

        # 4. 尾部卷积
        self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

    def _make_upsample_module(self, n_feats, scale):
        """恢复原版上采样：PixelShuffle堆叠 + ReLU激活，无GN"""
        m = []
        if (scale & (scale - 1)) == 0:  # 仅支持2的幂次上采样（原版EDSR设计）
            for _ in range(int(torch.log2(torch.tensor(scale)))):
                conv = nn.Conv2d(n_feats, n_feats * 4, 3, padding=1)
                # 恢复原版初始化强度（1/n_feats）
                nn.init.constant_(conv.weight, 1.0 / n_feats)
                nn.init.zeros_(conv.bias)
                m.append(conv)
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace=True))  # 恢复ReLU激活
        else:
            # 非2的幂次上采样：保持原版插值逻辑
            m.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
            m.append(nn.ReLU(inplace=True))
        return nn.Sequential(*m)

    def forward(self, x):
        # 预处理
        x = self.sub_mean(x)
        
        # 头部特征提取
        x_head = self.head(x)
        
        # 残差块特征提取 + 全局残差连接
        res = self.body(x_head)
        res = self.body_conv(res)
        x_body = x_head + res
        
        # 上采样（原版PixelShuffle）
        x_up = self.upsample(x_body)
        
        # 尾部卷积
        x_out = self.tail(x_up)
        x_out = self.add_mean(x_out)
        
        return x_out

# ==============================================================================
# 测试代码（验证原版EDSR可运行+维度正确）
# ==============================================================================
if __name__ == "__main__":
    # 初始化模型：16个残差块，64维特征，2倍超分（原版EDSR标准配置）
    model = EDSR(n_resblocks=16, n_feats=64, scale=2)
    # 构造测试输入：[batch_size, channel, height, width]
    test_input = torch.randn(1, 1, 512, 512)
    # 模型前向传播
    test_output = model(test_input)
    # 打印尺寸信息
    print(f"EDSR - 输入尺寸：{test_input.shape} | 输出尺寸：{test_output.shape}")
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EDSR - 总参数量：{total_params / 1000:.2f}k")