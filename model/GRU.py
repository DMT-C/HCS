import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.total_channels = input_channels + hidden_channels

        # Reset gate: 决定保留多少前一时刻信息
        self.reset_gate = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        # Update gate: Wz * xt + Uz * h_(t-1) for sigma
        self.update_gate = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        # Candidate hidden state
        self.candidate = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        # 权重初始化（Xavier均匀分布）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h_prev=None, M=None):
        # x: (batch, input_channels, height, width)
        # h_prev: (batch, hidden_channels, height, width)，若None则初始化为0
        # M: (input_channels, hidden_channels)，二维矩阵调制因子，若None则默认调制为1
        batch, _, height, width = x.size()
        
        if h_prev is None:
            h_prev = torch.zeros(batch, self.hidden_channels, height, width, 
                               device=x.device, dtype=x.dtype)
        
        # 拼接输入和前一隐藏状态
        combined = torch.cat([x, h_prev], dim=1)  # (batch, total_channels, height, width)

        # 计算重置门
        reset = torch.sigmoid(self.reset_gate(combined))  # (batch, hidden_channels, height, width)

        # 计算更新门：σ(Wz * xt + Uz * h_(t-1)) ⊙ tanh(x_t^T M h_(t-1))
        update_linear = self.update_gate(combined)  # (batch, hidden_channels, height, width)
        sigmoid_part = torch.sigmoid(update_linear)  # (batch, hidden_channels, height, width)

        # 计算 x_t^T M h_(t-1)
                # 替换原代码中从x_flat到xmh_diag的部分
        if M is not None:
            M = M.to(device=x.device, dtype=x.dtype)  # (input_channels, hidden_channels)
            x_flat = x.permute(0, 2, 3, 1).reshape(batch * height * width, self.input_channels)  # (b*h*w, input_channels)
            h_flat = h_prev.permute(0, 2, 3, 1).reshape(batch * height * width, self.hidden_channels)  # (b*h*w, hidden_channels)
            
            # 优化：直接计算每个位置的x^T M h（即对角线元素），避免完整矩阵乘法
            xm = torch.matmul(x_flat, M)  # (b*h*w, hidden_channels)
            # 逐行计算内积（等价于取矩阵乘法的对角线）
            xmh_diag = torch.sum(xm * h_flat, dim=1)  # (b*h*w,)  <-- 关键优化
            xmh_diag = xmh_diag.reshape(batch, height, width, 1)  # (batch, h, w, 1)
            modulation = torch.tanh(xmh_diag).permute(0, 3, 1, 2)  # (batch, 1, h, w)
        else:
            modulation = torch.ones(batch, 1, height, width, device=x.device, dtype=x.dtype)

        # 调制更新门
        update = sigmoid_part * modulation  # Broadcasting: (batch, hidden_channels, h, w) * (batch, 1, h, w)

        # 候选状态（仅用重置后的前隐藏）
        combined_reset = torch.cat([x, reset * h_prev], dim=1)
        candidate = torch.tanh(self.candidate(combined_reset))  # (batch, hidden_channels, height, width)

        # 新隐藏状态：渐变更新
        h_next = (1 - update) * h_prev + update * candidate

        return h_next

# 示例使用
if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)  # batch=2, channels=3, 64x64图像
    h_prev = torch.randn(2, 3, 64, 64)  # 前隐藏状态
    M = torch.randn(3, 3)  # M: (input_channels=3, hidden_channels=16)
    cell = ConvGRUCell(input_channels=3, hidden_channels=3)
    h_next = cell(x, h_prev, M)
    print(f"输出形状: {h_next.shape}")  # (2, 16, 64, 64)
    
    # 测试默认M=None
    h_next_default = cell(x, h_prev)
    print(f"默认M输出形状: {h_next_default.shape}")