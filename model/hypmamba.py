import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from hyptorch import pmath

class HyperbolicLinear(nn.Module):
    """双曲线性层，基于 Möbius 矩阵乘法"""
    def __init__(self, in_features, out_features, c, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(pmath.project(torch.randn(out_features, in_features, **factory_kwargs) * 0.01, c=c))
        if bias:
            self.bias = nn.Parameter(pmath.project(torch.zeros(out_features, **factory_kwargs), c=c))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = pmath.mobius_matvec(self.weight, x, c=self.c)
        if self.bias is not None:
            x = pmath.mobius_add(x, self.bias, c=self.c)
        x = pmath.project(x, c=self.c)
        return x

class HyperbolicConv1d(nn.Module):
    """双曲 1D 卷积，基于 Möbius 矩阵乘法"""
    def __init__(self, in_channels, out_channels, kernel_size, c, groups=1, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.c = c
        # 卷积核：(out_channels, in_channels//groups, kernel_size)
        self.weight = nn.Parameter(pmath.project(torch.randn(out_channels, in_channels//groups, kernel_size, **factory_kwargs) * 0.01, c=c))
        if bias:
            self.bias = nn.Parameter(pmath.project(torch.zeros(out_channels, **factory_kwargs), c=c))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B, in_channels, L), in Poincaré Ball
        B, C, L = x.shape
        out = torch.zeros(B, self.out_channels, L, device=x.device, dtype=x.dtype)
        for t in range(L):
            start = max(0, t - self.kernel_size + 1)
            x_window = x[:, :, start:t+1]  # (B, in_channels, window_size)
            # 逐点 Möbius 矩阵乘法
            for g in range(self.groups):
                in_start = g * (self.in_channels // self.groups)
                out_start = g * (self.out_channels // self.groups)
                x_g = x_window[:, in_start:in_start + self.in_channels//self.groups]  # (B, in_channels//groups, window_size)
                w_g = self.weight[out_start:out_start + self.out_channels//self.groups]  # (out_channels//groups, in_channels//groups, kernel_size)
                for i in range(x_g.shape[-1]):
                    x_t = x_g[:, :, i]  # (B, in_channels//groups)
                    w_t = w_g[:, :, min(i, self.kernel_size-1)]  # (out_channels//groups, in_channels//groups)
                    out[:, out_start:out_start + self.out_channels//self.groups, t] += pmath.mobius_matvec(w_t, x_t, c=self.c)
            if self.bias is not None:
                out[:, :, t] = pmath.mobius_add(out[:, :, t], self.bias, c=self.c)
            out[:, :, t] = pmath.project(out[:, :, t], c=self.c)
        return out

def hyperbolic_selective_scan_fn(x, dt, A, B, C, D, delta_bias, delta_softplus=True, c=0.7):
    """
    双曲选择性扫描函数
    Args:
        x: (B, d_inner//2, L), input in Poincaré Ball
        dt: (B, d_inner//2, L), time step in Poincaré Ball
        A: (d_inner//2, d_state), state matrix in Poincaré Ball
        B: (B, d_state, L), input matrix in Poincaré Ball
        C: (B, d_state, L), output matrix in Poincaré Ball
        D: (d_inner//2,), skip connection in Poincaré Ball
        delta_bias: (d_inner//2,), bias for dt
        delta_softplus: bool, apply softplus to dt
        c: float, curvature
    Returns:
        y: (B, d_inner//2, L), output in Poincaré Ball
    """
    b, d, L = x.shape
    _, d_state, _ = B.shape
    y = torch.zeros_like(x)
    h = pmath.project(torch.zeros(b, d, d_state, device=x.device, dtype=x.dtype), c=c)

    if delta_softplus:
        dt = F.softplus(dt + delta_bias.unsqueeze(-1))

    for t in range(L):
        x_t = x[:, :, t]
        dt_t = dt[:, :, t]
        B_t = B[:, :, t]
        C_t = C[:, :, t]

        Ah = pmath.mobius_matvec(A, h, c=c)
        Bx = pmath.mobius_matvec(B_t.unsqueeze(1), x_t.unsqueeze(-1), c=c).squeeze(-1)
        h = pmath.mobius_add(Ah, Bx.unsqueeze(-1) * dt_t.unsqueeze(-1).unsqueeze(-1), c=c)
        h = pmath.project(h, c=c)

        y_t = pmath.mobius_matvec(C_t.unsqueeze(1), h, c=c).squeeze(-1)
        y_t = pmath.mobius_pointwise_mul(y_t, D, c=c)
        y_t = pmath.project(y_t, c=c)
        y[:, :, t] = y_t

    return y

class HyperbolicMambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        c=0.7,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.c = c

        # 输入投影
        self.in_proj = HyperbolicLinear(d_model, self.d_inner, c=c, bias=bias, **factory_kwargs)

        # 动态参数生成
        self.x_proj = HyperbolicLinear(self.d_inner//2, self.dt_rank + self.d_state * 2, c=c, bias=False, **factory_kwargs)

        # 时间步投影
        self.dt_proj = HyperbolicLinear(self.dt_rank, self.d_inner//2, c=c, bias=True, **factory_kwargs)

        # 时间步参数初始化
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(pmath.project(inv_dt, c=c))
        self.dt_proj.bias._no_reinit = True

        # 状态矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(pmath.project(A_log, c=c))
        self.A_log._no_weight_decay = True

        # 跳跃连接
        self.D = nn.Parameter(pmath.project(torch.ones(self.d_inner//2, device=device), c=c))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = HyperbolicLinear(self.d_inner, d_model, c=c, bias=bias, **factory_kwargs)

        # 双曲卷积
        self.conv1d_x = HyperbolicConv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            kernel_size=d_conv,
            c=c,
            groups=self.d_inner//2,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.conv1d_z = HyperbolicConv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            kernel_size=d_conv,
            c=c,
            groups=self.d_inner//2,
            bias=conv_bias,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D), in Poincaré Ball
        Returns: (B, L, D), in Poincaré Ball
        """
        _, seqlen, _ = hidden_states.shape

        # 输入投影
        xz = self.in_proj(hidden_states)  # (B, L, d_inner)

        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  # x, z: (B, d_inner//2, L)

        # 双曲卷积
        x = self.conv1d_x(x)  # (B, d_inner//2, L)
        z = self.conv1d_z(z)  # (B, d_inner//2, L)
        x = pmath.project(F.silu(x), c=self.c)  # 双曲激活
        z = pmath.project(F.silu(z), c=self.c)

        # 动态参数生成
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (B*L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 时间步投影
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)  # (B, d_inner//2, L)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # 双曲选择性扫描
        y = hyperbolic_selective_scan_fn(
            x,
            dt,
            self.A_log,
            B,
            C,
            self.D,
            self.dt_proj.bias,
            delta_softplus=True,
            c=self.c
        )

        # 拼接与输出
        y = torch.cat([y, z], dim=1)  # (B, d_inner, L)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)  # (B, L, d_model)
        return out