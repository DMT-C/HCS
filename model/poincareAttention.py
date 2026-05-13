import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from geoopt import PoincareBall
from hyptorch.nn import HypLinear

# class Attention(nn.Module):

#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qk_norm=False,
#             attn_drop=0.,
#             c=0.7
#     ):
#         # 初始化类的构造函数
#         super().__init__()

#         # 确保维度可以被头数整除，以保证每个头的维度是整数
#         if dim<=3:
#             num_heads = 1
#         assert dim % num_heads == 0
#         self.c = c
#         self.poincare = PoincareBall(c=c)

#         # 初始化头的数量
#         self.num_heads = num_heads

#         # 计算每个头的维度
#         self.head_dim = dim // num_heads

#         # 初始化注意力的缩放因子
#         self.scale = self.head_dim ** -0.5



#         # 初始化注意力的dropout层
#         self.attn_drop = nn.Dropout(attn_drop)

#         # 初始化投影的线性层
#         self.proj = nn.Linear(dim, dim)
#     def mobius_add(self, x, y):
#         """Poincare球上的Möbius加法"""
#         return self.manifold.mobius_add(x, y)

#     def mobius_matvec(self, matrix, vector):
#         """Poincare球上的矩阵-向量乘法"""
#         return self.manifold.mobius_matvec(matrix, vector)
#     def forward(self, v,q,k):
#         #print("1 shape"+f"{x.shape}") 3,196,320
#         B, N, C = v.shape

#         q=q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k=k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v=v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)


        
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)

#         return x



class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qk_norm=False,
            attn_drop=0.,
            c=0.7,
            manifold=None,
    ):
        super().__init__()
        if dim <= 3:
            num_heads = 1
        assert dim % num_heads == 0
        self.c = c
        self.manifold = manifold if manifold is not None else PoincareBall(c=c)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        #self.scale = self.head_dim ** -0.5
        self.scale = torch.tensor(self.head_dim ** -0.5)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        self.proj = HypLinear(dim, dim,c=self.c)

    def mobius_add(self, x, y):
        return self.manifold.mobius_add(x, y)

    def mobius_matvec(self, matrix, vector):
        return self.manifold.mobius_matvec(matrix, vector)

    def batch_mobius_matvec(self, matrix, vector):
        """批量Poincare矩阵-向量乘法"""
        B, H, N, D = matrix.shape
        _, _, M, _ = vector.shape
        out = torch.zeros(B, H, N, M, device=matrix.device)
        for b in range(B):
            for h in range(H):
                out[b, h] = self.manifold.mobius_matvec(matrix[b, h], vector[b, h])
        return self.manifold.projx(out)

    def hyperbolic_softmax(self, scores, dim=-1):
        exp_scores = self.manifold.expmap0(scores)
        exp_sum = torch.sum(exp_scores, dim=dim, keepdim=True)
        weights = exp_scores / (exp_sum + 1e-10)
        return weights

    def forward(self, v, q, k):
        """
        v, q, k: 输入张量，形状 (B, N, C)，在Poincare球上
        返回: 输出张量，形状 (B, N, C)，在Poincare球上
        """
        B, N, C = v.shape

        # 确保输入在Poincare球上
        v = self.manifold.projx(v)
        q = self.manifold.projx(q)
        k = self.manifold.projx(k)

        # 重塑为多头格式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 缩放查询
        q = self.manifold.mobius_scalar_mul(self.scale, q)

        # 计算注意力分数
        q = q  
        k = k.transpose(-2, -1) 
        print(q.shape, k.shape)
        attn = self.batch_mobius_matvec(q, k)  

        # 双曲softmax
        attn = self.hyperbolic_softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 注意力加权
        x = self.batch_mobius_matvec(attn.unsqueeze(-1), v.unsqueeze(-2)).squeeze(-2)  # (B, num_heads, N, head_dim)

        # 重塑输出
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.manifold.projx(x)

        # 输出投影
        x = self.proj(x)
        return self.manifold.projx(x)