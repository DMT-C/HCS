import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = self.kernel_size

        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.gate_A = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_B = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        A_t = torch.sigmoid(self.gate_A(x))  # (B, out_C, H, W)
        B_t = torch.sigmoid(self.gate_B(x))  # (B, out_C, H, W)

        # 提取patch: (B, C*k*k, L)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        L = x_unfold.shape[-1]
        H_out = (H - kH) // kH + 1
        W_out = (W - kW) // kW + 1

        # Reshape为patch序列: (B, L, C, kH, kW)
        x_unfold = x_unfold.transpose(1, 2).contiguous().view(B, L, C, kH, kW)

        # 拉出门控参数 A, B 对应位置 (B, L, out_C, kH, kW)
        A_t_unfold = F.unfold(A_t, kernel_size=self.kernel_size, stride=self.stride).transpose(1, 2).contiguous()
        A_t_unfold = A_t_unfold.view(B, L, self.out_channels, kH, kW)

        B_t_unfold = F.unfold(B_t, kernel_size=self.kernel_size, stride=self.stride).transpose(1, 2).contiguous()
        B_t_unfold = B_t_unfold.view(B, L, self.out_channels, kH, kW)

        # 卷积核乘输入 patch 展平后投影: (B, L, out_C)
        kernel_exp = self.kernel.view(self.out_channels, self.in_channels, -1)  # (out_C, C, kH*kW)
        x_unfold_flat = x_unfold.view(B, L, self.in_channels, -1)  # (B, L, C, kH*kW)
        xi_out = torch.einsum("blci,oci->blo", x_unfold_flat, kernel_exp)  # (B, L, out_C)
        xi_out = self.norm(xi_out) * 0.1
        xi_out = xi_out.view(B, L, self.out_channels, 1, 1).expand(-1, -1, -1, kH, kW)

        # 初始化状态
        h = torch.zeros(B, self.out_channels, kH, kW, device=x.device)
        outputs = [None] * L

        # 蛇形扫描更新状态
        for row in range(H_out):
            start_idx = row * W_out
            end_idx = (row + 1) * W_out

            if row % 2 == 0:
                indices = list(range(start_idx, end_idx))  # 左到右
            else:
                indices = list(reversed(range(start_idx, end_idx)))  # 右到左

            for i in indices:
                Ai = A_t_unfold[:, i]      # (B, out_C, kH, kW)
                Bi = B_t_unfold[:, i]
                xi_i = xi_out[:, i]

                h = Ai * h + Bi * xi_i
                outputs[i] = h.unsqueeze(1)

        h_out = torch.cat(outputs, dim=1)  # (B, L, out_C, kH, kW)
        h_out = h_out.view(B, H_out, W_out, self.out_channels, kH, kW)
        h_out = h_out.permute(0, 3, 1, 4, 2, 5).contiguous()
        h_out = h_out.view(B, self.out_channels, H_out * kH, W_out * kW)

        return h_out


if __name__ == "__main__":
    x = torch.randn(2, 8, 12, 12)
    net = SSMConv2D(in_channels=8, out_channels=16, kernel_size=3)
    out = net(x)
    print(out.shape)  # e.g., torch.Size([2, 16, 12, 12])
