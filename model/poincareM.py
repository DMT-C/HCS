import torch
import torch.nn as nn
import torch.nn.functional as F

class PoincareMapper(nn.Module):
    """
    多通道 Poincaré 映射（各向异性通过马氏范数；不左乘 M，只改“尺子”）
    输入:
        x: [B,C,H,W]
    输出:
        映射向量 [B,C,H,W]
    """
    def __init__(self, num_channels, curvature=1.0, eps=1e-6, device="cuda:0"):
        super().__init__()
        self.register_buffer("C", torch.tensor(curvature, dtype=torch.float32, device=device))  # 随模型迁移设备
        self.eps = eps
        self.Cdim = num_channels
        self.device = device

        # 用下三角 L 参数化 SPD：M = L L^T + eps*I
        L_init = torch.randn(num_channels, num_channels, dtype=torch.float32, device=device)
        self.L_params = nn.Parameter(L_init)  # 学习参数

    def _spd_matrix(self):
        L = torch.tril(self.L_params)  # 下三角
        # 对角用 softplus 保证正
        diag = torch.diag_embed(F.softplus(torch.diag(L)) + 1e-6)
        L = L - torch.diag_embed(torch.diag(L)) + diag
        M = L @ L.transpose(-1, -2)
        M = M + self.eps * torch.eye(self.Cdim, device=M.device, dtype=M.dtype)
        #M=torch.eye(32).to(self.device)
        return M  # [C,C]

    @staticmethod
    def _safe_atanh(x, eps=1e-6):
        x = torch.clamp(x, min=-(1.0 - eps), max=(1.0 - eps))
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))  # atanh(x)

    def _mahalanobis_norm(self, x, M):
        """
        ||x||_M per pixel.
        x: [B,C,H,W], M: [C,C]
        return: [B,1,H,W]
        """
        x_chlast = x.permute(0, 2, 3, 1)                 # [B,H,W,C]
        quad = torch.einsum('bhwc,cd,bhwd->bhw', x_chlast, M, x_chlast)
        quad = torch.clamp(quad, min=0.0)                # 数值保护
        return torch.sqrt(quad + self.eps).unsqueeze(1)  # [B,1,H,W]

    def exp0(self, x,y):
        """
        指数映射 + 测地线因子 (共用 M)
        x: [B,C,H,W] 本时相
        y: [B,C,H,W] 另一时相
        """
        M = self._spd_matrix()
        normM = self._mahalanobis_norm(x, M)
        sqrtC = torch.sqrt(self.C)

        # 原始指数映射部分
        denom = sqrtC * normM
        scale = torch.tanh(denom) / torch.clamp(denom, min=self.eps)
        scale = torch.where(normM > self.eps, scale, torch.ones_like(normM))
        exp_base = x * scale

        #测地线距离因子 f(x,y)
        d_xy = self.distance(x, y)   # [B,1,H,W]
        factor =0.4+0.1* d_xy / (1.0 + d_xy)   # 0.4 0.1
        #factor=1
        exp_out = exp_base * factor

        return exp_out
    def log0(self, y):
        """
        对数映射（各向异性通过 ||y||_M）
        """
        M = self._spd_matrix()
        normM = self._mahalanobis_norm(y, M)
        sqrtC = torch.sqrt(self.C)

        arg = sqrtC * normM
        scale = self._safe_atanh(arg) / torch.clamp(arg, min=self.eps)
        scale = torch.where(normM > self.eps, scale, torch.ones_like(normM))

        return y * scale
    def get_M(self):
        """
        返回当前的 SPD 矩阵 M
        """
        return self._spd_matrix()


    def distance(self, u, v):

        M = self._spd_matrix()  # [C, C]

        # # 计算经过 M 调制的特征
        # uM = torch.einsum('bchw,cd->bdhw', u, M)  # u @ M
        # vM = torch.einsum('bchw,cd->bdhw', v, M)  # v @ M
        # 标准 Poincaré 距离（此处使用欧氏范数计算嵌入点的半径）
        ru2 = torch.sum((u) **2, dim=1, keepdim=True)    # [B,1,H,W]
        rv2 = torch.sum((v) **2, dim=1, keepdim=True)    # [B,1,H,W]
        diff2 = torch.sum((u - v)**2, dim=1, keepdim=True)  # [B,1,H,W]

        one = torch.tensor(1.0, device=u.device, dtype=u.dtype)
        denom = (one - self.C * ru2) * (one - self.C * rv2)
        denom = torch.clamp(denom, min=self.eps)

        arg = one + 2.0 * self.C * diff2 / denom
        arg = torch.clamp(arg, min=1.0 + 1e-9)  # acosh 输入需 ≥ 1
        return torch.acosh(arg)
    # def distance(self, u, v):
    #     """
    #     带有 SPD 调制的双曲距离: d_M(u, v)
    #     同时考虑 M 的各向异性与双曲几何特性
    #     u, v: [B, C, H, W]
    #     返回: [B, 1, H, W]
    #     """
    #     M = self._spd_matrix()  # [C, C]

    #     # 计算经过 M 调制的特征
    #     uM = torch.einsum('bchw,cd->bdhw', u, 0.1*M)  # u @ M
    #     vM = torch.einsum('bchw,cd->bdhw', v, 0.1*M)  # v @ M
 
    #     # 欧氏范数平方
    #     ru2 = torch.sum(uM ** 2, dim=1, keepdim=True)  # ||M u||^2
    #     rv2 = torch.sum(vM ** 2, dim=1, keepdim=True)  # ||M v||^2
    #     diff2 = torch.sum((uM - vM) ** 2, dim=1, keepdim=True)  # ||M(u - v)||^2

    #     one = torch.tensor(1.0, device=u.device, dtype=u.dtype)
    #     denom = (one - self.C * ru2) * (one - self.C * rv2)
    #     denom = torch.clamp(denom, min=self.eps)

    #     arg = one + 2.0 * self.C * diff2 / denom
    #     arg = torch.clamp(arg, min=1.0 + 1e-9)  # acosh 输入需 ≥ 1
    #     return torch.acosh(arg)
    
    def mobius_add(self, x, y):
        """
        Poincaré 球上的 Möbius 加法
        x, y: [B,C,H,W]
        返回: [B,C,H,W]
        """
        sqrtC = torch.sqrt(self.C)
        x2 = torch.sum(x**2, dim=1, keepdim=True)          # ||x||^2
        y2 = torch.sum(y**2, dim=1, keepdim=True)          # ||y||^2
        xy = torch.sum(x * y, dim=1, keepdim=True)         # <x,y>

        numerator = (1 + 2*self.C*xy + self.C*y2) * x + (1 - self.C*x2) * y
        denominator = 1 + 2*self.C*xy + (self.C**2) * x2 * y2
        denominator = torch.clamp(denominator, min=1e-6)

        return numerator / denominator

    def mobius_sub(self, x, y):
        """
        Poincaré 球上的 Möbius 减法
        x, y: [B,C,H,W]
        返回: [B,C,H,W]
        """
        return self.mobius_add(x, -y)


if __name__ == "__main__":
    torch.manual_seed(0)
    B,C,H,W = 1, 3, 32, 32
    x = torch.randn(B,C,H,W)
    x2=torch.randn(B,C,H,W)
    mapper = PoincareMapper(num_channels=C, curvature=2.0)
    print(x)
    y = mapper.exp0(x, x2)
    print(y)
    z = mapper.log0(y)
    print(x.shape, y.shape, z.shape)
    err = (z - x).abs().max().item()
    print("max |log0(exp0(x)) - x| =", err)
    d_self = mapper.distance(x, x).max().item()
    print("max distance(x,x) =", d_self)

