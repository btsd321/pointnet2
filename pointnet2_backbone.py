import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)
import torch
import torch.nn as nn
import pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG

class Pointnet2Backbone(nn.Module):
    r"""
        PointNet2骨干网络, 用于逐点特征提取(单尺度分组)。

        参数说明
        ----------
        npoint_per_layer: List[int], 长度为4
            每一层采样的点数
        radius_per_layer: List[float], 长度为4
            每一层分组的半径
        input_feature_dims: int = 0 
            每个点的特征描述符的输入通道数。如果点云为Nx9, 则该值应为6, 因为3个通道为xyz, 6个为特征描述符
        use_xyz: bool = True
            是否将xyz坐标作为特征输入
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        """
        初始化PointNet2骨干网络, 构建4层SA模块和4层FP模块。

        参数:
            npoint_per_layer: 每层采样点数列表
            radius_per_layer: 每层分组半径列表
            input_feature_dims: 输入特征通道数
            use_xyz: 是否使用xyz坐标作为特征
        """
        super(Pointnet2Backbone, self).__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[0],
                radius=radius_per_layer[0],
                nsample=32,
                mlp=[input_feature_dims, 32, 32, 64],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[1],
                radius=radius_per_layer[1],
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[2],
                radius=radius_per_layer[2],
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=npoint_per_layer[3],
                radius=radius_per_layer[3],
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))


    def _break_up_pc(self, pc):
        """
        将点云数据分解为xyz坐标和特征部分。

        参数:
            pc: 输入点云, 形状为(B, N, C)

        返回:
            xyz: (B, N, 3) 点的空间坐标
            features: (B, C-3, N) 点的特征(如果有)
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        网络前向传播

        参数:
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) 输入点云
                每个点格式为(x, y, z, features...)

        返回:
            new_features : torch.Tensor
                (B, 128, N) 逐点特征
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        # 依次通过4层SA模块, 逐步下采样和特征提取
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # 依次通过4层FP模块, 逐步特征上采样
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

class Pointnet2MSGBackbone(nn.Module):
    r"""
        PointNet2骨干网络, 用于逐点特征提取(多尺度分组)。

        参数说明
        ----------
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        """
        初始化PointNet2 MSG骨干网络, 构建4层MSG SA模块和4层FP模块。

        参数:
            npoint_per_layer: 每层采样点数列表
            radius_per_layer: 每层分组半径列表(每层为多尺度半径)
            input_feature_dims: 输入特征通道数
            use_xyz: 是否使用xyz坐标作为特征
        """
        super(Pointnet2MSGBackbone, self).__init__()
        assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.nscale = len(radius_per_layer[0])

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[0],
                radii=radius_per_layer[0],
                nsamples=[32]*self.nscale,
                mlps=[ [input_feature_dims, 32, 32, 64] for _ in range(self.nscale) ],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64*self.nscale
        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[1],
                radii=radius_per_layer[1],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 64, 64, 128] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128*self.nscale
        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[2],
                radii=radius_per_layer[2],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 128, 128, 256] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256*self.nscale
        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[3],
                radii=radius_per_layer[3],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 256, 256, 512] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512*self.nscale

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))


    def _break_up_pc(self, pc):
        """
        将点云数据分解为xyz坐标和特征部分。

        参数:
            pc: 输入点云, 形状为(B, N, C)

        返回:
            xyz: (B, N, 3) 点的空间坐标
            features: (B, C-3, N) 点的特征(如果有)
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        # xyz: batch * NumPoints * 3; features: batch* (num_dim-3) * NumPoints 
        return xyz, features

    def forward(self, pointcloud):
        """
        网络前向传播

        参数:
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) 输入点云
                每个点格式为(x, y, z, features...)

        返回:
            new_features : torch.Tensor
                (B, 128, N) 逐点特征
            Global_features : torch.Tensor
                最后一层全局特征
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        # 依次通过4层MSG SA模块, 逐步下采样和特征提取
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features) # l_features[-1] torch.Size([8, 1536, 64])
        Global_features = l_features[-1]
        # 依次通过4层FP模块, 逐步特征上采样
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0], Global_features

class Pointnet2MSGBackbone_simle(nn.Module):
    r"""
        PointNet2骨干网络(简化版), 用于逐点特征提取(多尺度分组)。

        参数说明
        ----------
    """

    def __init__(self, npoint_per_layer, radius_per_layer, input_feature_dims=0, use_xyz=True):
        """
        初始化简化版PointNet2 MSG骨干网络, 仅包含2层MSG SA模块和2层FP模块。

        参数:
            npoint_per_layer: 每层采样点数列表
            radius_per_layer: 每层分组半径列表(每层为多尺度半径)
            input_feature_dims: 输入特征通道数
            use_xyz: 是否使用xyz坐标作为特征
        """
        super(Pointnet2MSGBackbone_simle, self).__init__()
        # assert len(npoint_per_layer) == len(radius_per_layer) == 4
        self.nscale = len(radius_per_layer[0])

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[0], # 最远点采样
                radii=radius_per_layer[0],
                nsamples=[32]*self.nscale,
                mlps=[ [input_feature_dims, 32, 32, 64] for _ in range(self.nscale) ],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64*self.nscale
        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=npoint_per_layer[1],
                radii=radius_per_layer[1],
                nsamples=[32]*self.nscale,
                mlps=[ [c_in, 64, 64, 128] for _ in range(self.nscale) ] ,
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128*self.nscale
        # 后续层如需扩展可参考注释部分

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_feature_dims, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_1 + c_out_0, 256, 128]))
        # 后续层如需扩展可参考注释部分

    def _break_up_pc(self, pc):
        """
        将点云数据分解为xyz坐标和特征部分。

        参数:
            pc: 输入点云, 形状为(B, N, C)

        返回:
            xyz: (B, N, 3) 点的空间坐标
            features: (B, C-3, N) 点的特征(如果有)
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        """
        网络前向传播

        参数:
            pointcloud: torch.cuda.FloatTensor
                (B, N, 3 + input_channels) 输入点云
                每个点格式为(x, y, z, features...)

        返回:
            new_features : torch.Tensor
                (B, 128, N) 逐点特征
            Global_features : torch.Tensor
                最后一层全局特征
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        # 依次通过2层MSG SA模块, 逐步下采样和特征提取
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features) # l_features[-1] torch.Size([8, 1536, 64])
        Global_features = l_features[-1]
        # 依次通过2层FP模块, 逐步特征上采样
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0], Global_features

if __name__ == "__main__":
    # 测试主程序
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = torch.randn(8, 16384, 3).cuda()
    xyz.requires_grad=True

    test_module = Pointnet2Backbone([4096,1024,256,64], [30,60,120,240])
    test_module.cuda()
    # print(test_module(xyz))


    for _ in range(1):
        new_features = test_module(xyz)
        print('new_features', new_features.shape)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        # print(new_features)
        print(xyz.grad)
