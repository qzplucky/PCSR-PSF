import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, res_scale=0.1, dropout_rate=0.05):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2)
        self.res_scale = res_scale
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        res = self.conv1(x)
        res = self.gn(res)
        res = self.act(res)
        res = self.dropout(res)
        res = self.conv2(res)
        return x + res * self.res_scale

class EDSR_Lite(nn.Module):
    def __init__(self, n_resblocks=8, n_feats=64, scale=2, dropout_rate=0.05):
        super().__init__()
        self.scale = scale
        self.n_feats = n_feats
        self.sub_mean = nn.Identity()
        self.add_mean = nn.Identity()

        self.head = nn.Conv2d(1, n_feats, 3, padding=1)

        self.body = nn.Sequential(*[
            ResBlock(n_feats, dropout_rate=dropout_rate) 
            for _ in range(n_resblocks)
        ])
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.body_dropout = nn.Dropout2d(p=dropout_rate)

        self.upsample = self._make_upsample_module(n_feats, scale)

        self.skip_fusion = nn.Conv2d(n_feats * 2, n_feats, 3, padding=1)

        self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

    def _make_upsample_module(self, n_feats, scale):
        m = []
        if scale == 2:
            conv_ps = nn.Conv2d(n_feats, n_feats * scale ** 2, 3, padding=1)
            nn.init.constant_(conv_ps.weight, 1.0 / (n_feats * 10))
            nn.init.zeros_(conv_ps.bias)
            m.append(conv_ps)
            m.append(nn.PixelShuffle(scale))
            m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
            
        elif scale == 4:
            conv_ps = nn.Conv2d(n_feats, n_feats * 4, 3, padding=1)
            nn.init.constant_(conv_ps.weight, 1.0 / (n_feats * 10))
            nn.init.zeros_(conv_ps.bias)
            m.append(conv_ps)
            m.append(nn.PixelShuffle(2))
            m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
            m.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        else:
            m.append(nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))
            m.append(nn.GroupNorm(num_groups=8, num_channels=n_feats, eps=1e-6))
            m.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        return nn.Sequential(*m)

    def forward(self, x):
        x = self.sub_mean(x)
        
        x_head = self.head(x)
        
        res = self.body(x_head)
        res = self.body_conv(res)
        res = self.body_dropout(res)
        x_body = x_head + res
        
        x_up = self.upsample(x_body)
        
        x_head_up = F.interpolate(x_head, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_fused = torch.cat([x_up, x_head_up], dim=1)
        x_fused = self.skip_fusion(x_fused)
        
        x_out = self.tail(x_fused)
        x_out = self.add_mean(x_out)
        
        return x_out

if __name__ == "__main__":
    model = EDSR_Lite(n_resblocks=8, n_feats=64, scale=2, dropout_rate=0.05)
    test_input = torch.randn(1, 1, 512, 512)
    test_output = model(test_input)
    print(f"EDSR-Lite - Input size: {test_input.shape} | Output size: {test_output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EDSR-Lite - Total parameters: {total_params / 1000:.2f}k")
