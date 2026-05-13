import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

def JS_loss(p, q, get_softmax=True):
    softmax2d = nn.Softmax2d()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p = softmax2d(p)
        q = softmax2d(q)
        # p = F.softmax(p)
        # q = F.softmax(q)
    leg_mean = ((p + q) / 2).log()
    return (KLDivLoss(leg_mean, p) + KLDivLoss(leg_mean, q)) / 2


def Cosine(x, y):
    xx = torch.sum(x ** 2, dim=1) ** 0.5
    x = x / xx[:, np.newaxis]

    yy = torch.sum(y ** 2, dim=1) ** 0.5
    y = y / torch.unsqueeze(yy, dim=1)

    dist = 1-torch.dot(x, y.transpose())
    return dist

def l2_distance(x, y):
    l2_dis = torch.sum(((x - y) ** 2), dim=1)
    return l2_dis

def l2_regularization(model,lambda_reg=0.00005):

    l2_reg = torch.tensor(0.0,device=torch.device('cuda'))
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2)
    return lambda_reg * l2_reg

def spectral_consistency_loss(z1, z2, M, eps=1e-6):
    """
    光谱一致性损失
    z1, z2: [B,C,H,W] 两个时相的映射结果
    M: [C,C] SPD 矩阵
    """
    # (x-y)^T M (x-y) 逐像素马氏距离
    diff = z1 - z2                       # [B,C,H,W]
    diff_chlast = diff.permute(0, 2, 3, 1)  # [B,H,W,C]
    quad = torch.einsum('bhwc,cd,bhwd->bhw', diff_chlast, M, diff_chlast)
    quad = torch.clamp(quad, min=0.0)
    return quad.mean()
def complementarity_loss(h1, h2):
    """
    互补性损失
    h1, h2: [B,C] 或 [B,C,H,W] 的 hidden state
    """
    # 拉平 spatial 维度 -> [B, C, N]
    if h1.dim() == 4:  
        B,C,H,W = h1.shape
        h1 = h1.view(B,C,-1)
        h2 = h2.view(B,C,-1)

    # 余弦相似度 [B,N]
    cos_sim = F.cosine_similarity(h1, h2, dim=1)
    return cos_sim.mean()  # 越大越相似 -> loss 越大


import torch
import torch.nn as nn
from geoopt import PoincareBall

class PoincareSimilarityLoss(nn.Module):
    def __init__(self, c=0.7):
        super().__init__()
        self.poincare = PoincareBall(c=c)

    def forward(self, dist_map1, dist_map2):
        """
        基于 Poincaré 距离计算相似性损失。
        输入：dist_map1, dist_map2 [batch_size, poincare_dim, H, W]
        输出：标量损失
        """
        batch_size, _, H, W = dist_map1.shape
        dist_map1 = dist_map1.permute(0, 2, 3, 1).reshape(-1, 2)  # [batch_size*H*W, 2]
        dist_map2 = dist_map2.permute(0, 2, 3, 1).reshape(-1, 2)  # [batch_size*H*W, 2]
        distance = self.poincare.dist(dist_map1, dist_map2)  # [batch_size*H*W]
        d_min = torch.min(distance)
        d_max = torch.max(distance)
        if d_max == d_min:
            print("Warning: distance max equals min, setting normalized distance to zeros")
            distance_normalized = torch.zeros_like(distance)
        else:
            distance_normalized = (distance - d_min) / (d_max - d_min + 1e-10)
        
        # 求均值作为损失
        loss = distance_normalized.mean()

        return loss