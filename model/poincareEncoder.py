import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mamba_ssm import Mamba
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from model.poincareConv import SSMConv2D
from geoopt import PoincareBall
import geoopt
from hyptorch.nn import HypLinear,ToPoincare,FromPoincare
from model.net import Encoder,Decoder
from model.KAN import KANLinear
from model.poincareM import PoincareMapper
from model.GRU import ConvGRUCell


class MambaScan(nn.Module):
    def __init__(
        self,
        # 模型的维度配置
        d_model,
        # 状态向量的维度
        d_state=16,
        # 卷积层的维度
        d_conv=4,
        # 扩张率，用于空洞卷积
        expand=2,
        # 时间步长的排名，"auto"表示自动选择
        dt_rank="auto",
        # 时间步长的最小值
        dt_min=0.001,
        # 时间步长的最大值
        dt_max=0.1,
        # 时间步长的初始化方法
        dt_init="random",
        # 时间步长的缩放因子
        dt_scale=1.0,
        # 时间步长初始化的下限
        dt_init_floor=1e-4,
        # 是否在卷积层中使用偏置
        conv_bias=True,
        # 是否在全连接层中使用偏置
        bias=False,
        # 是否使用快速路径以加速计算
        use_fast_path=True,
        # 层索引，用于标识特定的层
        layer_idx=None,
        # 设备类型，如"cpu"或"cuda"
        device=None,
        # 数据类型，如torch.float32
        dtype=None,
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
        # 输入投影
        #self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        # 动态参数生成
        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # 时间步投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)
        # 时间步参数初始化
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # dt初始化
        dt = torch.exp(
            torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # 稳定计算
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt) # 设置偏置为inv_dt
        self.dt_proj.bias._no_reinit = True   # 锁定优化器更新
        # 状态矩阵生成
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        # 参数化
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True # 不参与权重衰减
        # 跳跃连接参数
        self.D = nn.Parameter(torch.ones(self.d_model, device=device))
        self.D._no_weight_decay = True

        # 分组卷积
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_model,
            **factory_kwargs,
        )
        self.conv1d_y = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_model,
            **factory_kwargs,
        )

        

    def forward(self, x,y):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(f"1{hidden_states.shape}")
        _, seqlen, _ = x.shape
        x = rearrange(x, "b l d -> b d l")
        y=rearrange(y, "b l d -> b d l")
        A = -torch.exp(self.A_log.float()) # 从对数空间提取出来
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_model))
        y=F.silu(F.conv1d(input=y, weight=self.conv1d_y.weight, bias=self.conv1d_y.bias, padding='same', groups=self.d_model))
        y_dbl = self.x_proj(rearrange(y, "b d l -> (b l) d"))

        dt, B, C = torch.split(y_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        

        y = rearrange(y, "b d l -> b l d")

        # print(f"2{out.shape}")
        return y



class SpiralScanModule(nn.Module):
    def __init__(self, hidden_channels=32,kernel_size=3,stride=2,padding=1):
        super().__init__()      
        self.hidden_channels=hidden_channels
        self.mamba = MambaScan(d_model=hidden_channels, d_state=hidden_channels)
        self.GRU=ConvGRUCell(input_channels=hidden_channels, hidden_channels=hidden_channels, kernel_size=3, padding=1)
        self.h=None
        # self.downsample_in=nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, stride=stride,kernel_size=kernel_size, padding=padding, bias=False)
        # self.upsample_out = nn.ConvTranspose2d(
        #     in_channels=hidden_channels,
        #     out_channels=hidden_channels,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        # )
    def get_h(self,):
        return self.h   
    def _mamba_scan(self,x,y, batch_size, H, W,M):
        """
        Mamba-style scanning with local/global/merged features as B/C/A.
        Input: local_p, global_p, merged_p [batch_size, poincare_dim=2, H, W]
        Output: dist_map [batch_size, spiral_steps, H, W]
        """
        
        # y=self.downsample_in(y)
        # x=self.downsample_in(x)
        # _,_,dh,dw=y.shape
        y=self.GRU(x,h_prev=y,M=M)
        #y=x
        self.h=y
        x = x.permute(0, 2, 3, 1).reshape(1, -1, self.hidden_channels)
        y= y.permute(0, 2, 3, 1).reshape(1, -1, self.hidden_channels)
        out = self.mamba(x,y)  
        out = out.reshape(batch_size, H, W,self.hidden_channels).permute(0,3,1,2)  # [batch_size, d, chunk, chunk]
        #out=self.upsample_out(out).reshape(batch_size, -1, H, W)
        return out
    # def _mamba_scan(self,x,y, batch_size, H, W,M):
    #     """
    #     Mamba-style scanning with local/global/merged features as B/C/A.
    #     Input: local_p, global_p, merged_p [batch_size, poincare_dim=2, H, W]
    #     Output: dist_map [batch_size, spiral_steps, H, W]
    #     """
    #     y=self.GRU(x,h_prev=y,M=M)
    #     self.h=y
    #     x = x.permute(0, 2, 3, 1).reshape(batch_size, H*W, -1)
    #     y= y.permute(0, 2, 3, 1).reshape(batch_size, H*W, -1)
    #     out = self.mamba(x,y)  
    #     out = out.view(batch_size, H, W, -1).permute(0, 3, 1, 2)  # [batch_size, d, chunk, chunk]
    #     return out
 
    def forward(self,x,y,M):

        batch_size, c, H, W = x.shape
        out1= self._mamba_scan(x,y,batch_size, H, W,M)
        return  out1




class PoincareEncoder(nn.Module):
    def __init__(self, in_channels=3,out_channels=32, c=0.7,kernel_size=3):
        super().__init__()
        self.poincareMapper = PoincareMapper(num_channels=out_channels,curvature=c)
        self.projector = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.encoder = Encoder(in_channels=out_channels, out_channels=out_channels)
        self.spiral_scan1 = SpiralScanModule(hidden_channels=out_channels, kernel_size=kernel_size)
        #self.spiral_scan2 = SpiralScanModule(hidden_channels=out_channels, kernel_size=kernel_size)


    def get_M(self,):
        return self.poincareMapper.get_M()
    def get_h(self,):
        return self.spiral_scan1.get_h()

    def forward(self, img1, img2):
        feature1=self.projector(
            img1
        )
        feature2=self.projector(
            img2
        )
        feature1=self.poincareMapper.exp0(feature1,feature2)
        feature2=self.poincareMapper.exp0(feature2,feature1)
        M=self.poincareMapper.get_M()
        #M=torch.eye(32)

        out1= self.spiral_scan1(
            feature1,feature2,M
        )
        h1=self.spiral_scan1.get_h()
        out2= self.spiral_scan1(
            feature2,feature1,M
        )
        h2=self.spiral_scan1.get_h()
        # out1_a= self.spiral_scan1(
        #     feature1,feature2,M
        # )
        # h1=self.spiral_scan1.get_h()
        # out2_a= self.spiral_scan1(
        #     feature2,feature1,M
        # )
        # h2=self.spiral_scan1.get_h()


 
        return out1,out2,feature1,feature2,h1,h2
        #return out1,out2,None,None,None,None


class PoincareDecoder(nn.Module):
    def __init__(self, in_channels=32,out_channels=3, c=0.7,kernel_size=3):
        super().__init__()
        self.spiral_scan1 = SpiralScanModule(hidden_channels=in_channels, kernel_size=kernel_size)
 
        self.decoder = Decoder(in_channels=in_channels, out_channels=in_channels)
        self.poincareMapper = PoincareMapper(num_channels=in_channels,curvature=c)
        self.projector = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)


    def forward(self, img1):

        M=self.poincareMapper.get_M()
        # out= self.spiral_scan1(
        #     img1,img1,M
        # )
        #M=torch.eye(32)
        out= self.spiral_scan1(
            img1,img1,M
        )
  


        out=self.poincareMapper.log0(out)
        out= self.projector(out)

        return out


    