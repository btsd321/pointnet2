"""
PointNet2 backbone modules for point cloud processing.
"""

import os
import sys
import warnings

# 添加当前目录到 Python 路径
__file_dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__file_dir__)

# 导入主要的 PointNet2 模块
try:
    from .pointnet2_backbone import (
        Pointnet2Backbone,
        Pointnet2MSGBackbone,
        Pointnet2MSGBackbone_simle,
    )
    from .pointnet2_modules import (
        PointnetSAModule,
        PointnetSAModuleMSG,
        PointnetFPModule,
    )
    from .pointnet2_utils import *
    from .pytorch_utils import *
    
    __all__ = [
        'Pointnet2Backbone',
        'Pointnet2MSGBackbone',
        'Pointnet2MSGBackbone_simle',
        'PointnetSAModule',
        'PointnetSAModuleMSG',
        'PointnetFPModule',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import PointNet2 modules: {e}")
    __all__ = []
