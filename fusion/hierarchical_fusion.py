"""
层次化融合框架
Hierarchical Fusion Framework

逐步融合低级到高级特征，形成多层次的特征金字塔
"""

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch函数式接口，包含常用的神经网络操作
import torch.nn.functional as F
# 导入NumPy数值计算库
import numpy as np
# 导入类型提示模块，用于类型注解
from typing import Dict, List, Optional, Tuple


class FeaturePyramid(nn.Module):
    """特征金字塔"""

    def __init__(self,
                 input_dim: int,  # 输入特征的维度大小
                 feature_dims: List[int] = [64, 128, 256, 512],  # 各层特征维度的列表，默认为[64, 128, 256, 512]
                 dropout: float = 0.2):  # Dropout丢弃率，用于防止过拟合，默认为0.2
        """
        初始化特征金字塔

        Args:
            input_dim: 输入维度
            feature_dims: 各层特征维度
            dropout: Dropout率
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 计算并存储特征金字塔的层数，即feature_dims列表的长度
        self.num_levels = len(feature_dims)

        # 创建一个ModuleList用于存储各层的特征提取模块
        # 各层特征提取
        self.feature_layers = nn.ModuleList()
        # 初始化前一层的维度为输入维度
        prev_dim = input_dim
        # 遍历feature_dims列表，为每一层创建特征提取模块
        for dim in feature_dims:
            # 为当前层创建一个Sequential序列模块，包含线性层、层归一化、ReLU激活和Dropout
            self.feature_layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),  # 线性变换层，将前一层的维度映射到当前层的维度
                nn.LayerNorm(dim),  # 层归一化，对当前层的特征进行归一化处理
                nn.ReLU(),  # ReLU激活函数，引入非线性
                nn.Dropout(dropout)  # Dropout层，随机丢弃部分神经元以防止过拟合
            ))
            # 更新前一层的维度为当前层的维度，供下一层使用
            prev_dim = dim

        # 创建一个ModuleList用于存储自顶向下路径的模块
        # 自顶向下路径
        self.top_down = nn.ModuleList()
        # 从最高层向最低层遍历，创建自顶向下的连接
        for i in range(self.num_levels - 1, 0, -1):
            # 为每一对相邻层创建自顶向下的连接模块
            self.top_down.append(nn.Sequential(
                nn.Linear(feature_dims[i], feature_dims[i - 1]),  # 线性层，将高层特征维度映射到低层特征维度
                nn.LayerNorm(feature_dims[i - 1])  # 层归一化，对映射后的特征进行归一化
            ))

        # 创建一个ModuleList用于存储横向连接的模块
        # 横向连接
        self.lateral = nn.ModuleList()
        # 为每一层（除最高层外）创建横向连接模块
        for i in range(self.num_levels - 1):
            # 创建横向连接模块，保持特征维度不变
            self.lateral.append(nn.Sequential(
                nn.Linear(feature_dims[i], feature_dims[i]),  # 线性层，输入输出维度相同
                nn.LayerNorm(feature_dims[i])  # 层归一化
            ))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入特征 (batch, seq, input_dim)

        Returns:
            各层特征列表
        """
        # 创建一个空列表用于存储各层的特征
        # 自底向上
        features = []
        # 遍历所有特征提取层
        for layer in self.feature_layers:
            # 将输入特征通过当前层进行处理
            x = layer(x)
            # 将处理后的特征添加到特征列表中
            features.append(x)

        # 从倒数第二层开始向第一层遍历，进行自顶向下的特征融合
        # 自顶向下融合
        for i in range(self.num_levels - 2, -1, -1):
            # 获取自顶向下的特征，通过对应的top_down模块处理高层特征
            top_down_feat = self.top_down[self.num_levels - 2 - i](features[i + 1])
            # 获取横向连接的特征，通过对应的lateral模块处理当前层特征
            lateral_feat = self.lateral[i](features[i])
            # 将横向特征和自顶向下特征相加，实现特征融合
            features[i] = lateral_feat + top_down_feat

        # 返回融合后的各层特征列表
        return features


class HierarchicalFusionLayer(nn.Module):
    """层次化融合层"""

    def __init__(self,
                 input_dims: Dict[str, int],  # 各模态的输入维度字典，键为模态名称，值为维度大小
                 output_dim: int,  # 输出特征的维度大小
                 fusion_strategy: str = 'attention',  # 融合策略，可选'concat'、'add'、'attention'、'gated'，默认为'attention'
                 dropout: float = 0.2):  # Dropout丢弃率，默认为0.2
        """
        初始化层次化融合层

        Args:
            input_dims: 各模态输入维度
            output_dim: 输出维度
            fusion_strategy: 融合策略 ('concat', 'add', 'attention', 'gated')
            dropout: Dropout率
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 获取所有模态的名称列表
        self.modalities = list(input_dims.keys())
        # 存储融合策略
        self.fusion_strategy = fusion_strategy

        # 创建一个ModuleDict用于存储各模态的投影层
        # 模态投影
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, output_dim)  # 为每个模态创建一个线性投影层，将其维度映射到统一的输出维度
            for mod, dim in input_dims.items()  # 遍历所有模态及其维度
        })

        # 根据融合策略创建相应的融合模块
        # 融合策略
        if fusion_strategy == 'concat':
            # 如果是拼接策略，创建一个线性层将拼接后的特征映射到输出维度
            self.fusion = nn.Linear(output_dim * len(self.modalities), output_dim)
        elif fusion_strategy == 'attention':
            # 如果是注意力策略，创建多头注意力层
            self.attention = nn.MultiheadAttention(output_dim, 4, dropout=dropout, batch_first=True)
            # 创建一个线性层用于最终的融合
            self.fusion = nn.Linear(output_dim, output_dim)
        elif fusion_strategy == 'gated':
            # 如果是门控策略，为每个模态创建一个门控机制
            self.gates = nn.ModuleDict({
                mod: nn.Sequential(
                    nn.Linear(output_dim, output_dim),  # 线性层
                    nn.Sigmoid()  # Sigmoid激活函数，输出0-1之间的门控值
                )
                for mod in self.modalities  # 遍历所有模态
            })
            # 创建一个线性层用于最终的融合
            self.fusion = nn.Linear(output_dim, output_dim)

        # 创建层归一化层，用于稳定训练
        self.layer_norm = nn.LayerNorm(output_dim)
        # 创建Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: 各模态输入 {modality: (batch, seq, dim)}

        Returns:
            融合特征 (batch, seq, output_dim)
        """
        # 创建一个字典用于存储投影后的各模态特征
        # 投影
        projected = {}
        # 遍历所有模态
        for mod in self.modalities:
            # 检查当前模态是否在输入中存在
            if mod in inputs:
                # 将当前模态的特征通过对应的投影层进行投影
                projected[mod] = self.projections[mod](inputs[mod])

        # 根据融合策略进行特征融合
        # 融合
        if self.fusion_strategy == 'concat':
            # 如果是拼接策略，将所有模态的特征在最后一个维度上进行拼接
            fused = torch.cat([projected[mod] for mod in self.modalities if mod in projected], dim=-1)
            # 通过线性层将拼接后的特征映射到输出维度
            fused = self.fusion(fused)

        elif self.fusion_strategy == 'add':
            # 如果是加法策略，直接将所有模态的特征相加
            fused = sum(projected.values())

        elif self.fusion_strategy == 'attention':
            # 如果是注意力策略，简化处理：直接使用加法融合
            # 简化注意力融合：直接使用add
            fused = sum(projected.values())
            # 通过线性层进行最终融合
            fused = self.fusion(fused)

        elif self.fusion_strategy == 'gated':
            # 如果是门控策略，创建一个列表存储门控后的特征
            gated_features = []
            # 遍历所有模态
            for mod in self.modalities:
                # 检查当前模态是否在投影后的特征中存在
                if mod in projected:
                    # 通过门控机制计算门控值
                    gate = self.gates[mod](projected[mod])
                    # 将原始特征与门控值相乘，实现自适应加权
                    gated_features.append(projected[mod] * gate)
            # 将所有门控后的特征相加
            fused = sum(gated_features)
            # 通过线性层进行最终融合
            fused = self.fusion(fused)

        # 对融合后的特征进行层归一化
        fused = self.layer_norm(fused)
        # 应用Dropout
        fused = self.dropout(fused)

        # 返回融合后的特征
        return fused


class HierarchicalFusionNetwork(nn.Module):
    """层次化融合网络"""

    def __init__(self,
                 modalities: List[str],  # 模态名称列表
                 input_dims: Dict[str, int],  # 各模态的输入维度字典
                 feature_dims: List[int] = [64, 128, 256, 512],  # 各层特征维度的列表，默认为[64, 128, 256, 512]
                 fusion_strategy: str = 'attention',  # 融合策略，默认为'attention'
                 dropout: float = 0.2,  # Dropout丢弃率，默认为0.2
                 use_skip_connections: bool = True):  # 是否使用跳跃连接，默认为True
        """
        初始化层次化融合网络

        Args:
            modalities: 模态列表
            input_dims: 各模态输入维度
            feature_dims: 各层特征维度
            fusion_strategy: 融合策略
            dropout: Dropout率
            use_skip_connections: 是否使用跳跃连接
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 存储模态列表
        self.modalities = modalities
        # 计算并存储特征金字塔的层数
        self.num_levels = len(feature_dims)
        # 存储是否使用跳跃连接的标志
        self.use_skip_connections = use_skip_connections

        # 为每个模态创建一个特征金字塔
        # 各模态特征金字塔
        self.pyramids = nn.ModuleDict({
            mod: FeaturePyramid(input_dims[mod], feature_dims, dropout)  # 创建特征金字塔实例
            for mod in modalities  # 遍历所有模态
        })

        # 创建一个ModuleList用于存储各层的融合模块
        # 各层融合
        self.fusion_layers = nn.ModuleList()
        # 遍历每一层，为每一层创建融合模块
        for i, dim in enumerate(feature_dims):
            # 创建当前层各模态的输入维度字典，所有模态在该层的维度相同
            layer_input_dims = {mod: dim for mod in modalities}
            # 创建层次化融合层并添加到列表中
            self.fusion_layers.append(
                HierarchicalFusionLayer(layer_input_dims, dim, fusion_strategy, dropout)
            )

        # 如果使用跳跃连接，创建跳跃连接模块
        # 跳跃连接
        if use_skip_connections:
            # 创建一个ModuleList用于存储跳跃连接模块
            self.skip_connections = nn.ModuleList()
            # 从第二层开始遍历，为每对相邻层创建跳跃连接
            for i in range(1, self.num_levels):
                # 创建一个线性层，将前一层的维度映射到当前层的维度
                self.skip_connections.append(
                    nn.Linear(feature_dims[i - 1], feature_dims[i])
                )

        # 创建输出模块，用于最终的特征处理
        # 输出
        self.output = nn.Sequential(
            nn.Linear(feature_dims[-1], feature_dims[-1]),  # 线性层，保持维度不变
            nn.LayerNorm(feature_dims[-1]),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout)  # Dropout层
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            inputs: 各模态输入

        Returns:
            输出字典
        """
        # 创建输出字典，初始化level_features为空列表
        outputs = {'level_features': []}

        # 创建一个字典用于存储各模态在各层的特征
        # 各模态特征金字塔
        modality_features = {}
        # 遍历所有模态
        for mod in self.modalities:
            # 检查当前模态是否在输入中存在
            if mod in inputs:
                # 将当前模态的输入通过对应的特征金字塔处理，得到各层的特征
                modality_features[mod] = self.pyramids[mod](inputs[mod])

        # 初始化前一层的融合特征为None
        # 层次化融合
        prev_fused = None
        # 遍历每一层进行层次化融合
        for level in range(self.num_levels):
            # 创建一个字典用于存储当前层各模态的特征
            # 获取当前层各模态特征
            level_inputs = {}
            # 遍历所有模态
            for mod in self.modalities:
                # 检查当前模态是否在特征字典中存在
                if mod in modality_features:
                    # 获取当前模态在当前层的特征
                    level_inputs[mod] = modality_features[mod][level]

            # 通过当前层的融合模块对当前层各模态特征进行融合
            # 融合
            fused = self.fusion_layers[level](level_inputs)

            # 如果使用跳跃连接且存在前一层的融合特征
            # 跳跃连接
            if self.use_skip_connections and prev_fused is not None:
                # 通过跳跃连接模块将前一层的融合特征映射到当前层的维度
                skip = self.skip_connections[level - 1](prev_fused)
                # 将跳跃连接的特征与当前层的融合特征相加
                fused = fused + skip

            # 更新前一层的融合特征为当前层的融合特征
            prev_fused = fused
            # 将当前层的融合特征添加到输出字典的level_features列表中
            outputs['level_features'].append(fused)

        # 通过输出模块对最终的融合特征进行处理
        # 最终输出
        outputs['output'] = self.output(prev_fused)

        # 返回输出字典
        return outputs


class ConsistencyLoss(nn.Module):
    """一致性损失（适配不同维度特征）"""

    def __init__(self,
                 loss_type: str = 'mse',  # 损失类型，可选'mse'、'kl'、'cosine'，默认为'mse'
                 temperature: float = 1.0):  # 温度参数，用于调节损失计算，默认为1.0
        """
        初始化一致性损失

        Args:
            loss_type: 损失类型 ('mse', 'kl', 'cosine')
            temperature: 温度参数
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 存储损失类型
        self.loss_type = loss_type
        # 存储温度参数
        self.temperature = temperature

        # 根据损失类型创建相应的损失函数
        if loss_type == 'mse':
            # 如果是MSE损失，创建均方误差损失函数
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'kl':
            # 如果是KL散度损失，创建KL散度损失函数
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif loss_type == 'cosine':
            # 如果是余弦相似度损失，创建余弦嵌入损失函数
            self.loss_fn = nn.CosineEmbeddingLoss()

        # 存储维度映射层（动态创建，避免初始化时未知维度）
        self.dim_mappers = nn.ModuleDict()

    def _get_dim_mapper(self, src_dim: int, tgt_dim: int) -> nn.Module:
        """获取/创建维度映射层（高维->低维）"""
        key = f"{src_dim}2{tgt_dim}"
        if key not in self.dim_mappers:
            self.dim_mappers[key] = nn.Sequential(
                nn.Linear(src_dim, tgt_dim),
                nn.LayerNorm(tgt_dim)
            )
        return self.dim_mappers[key]

    def forward(self,
                features1: torch.Tensor,  # 第一组特征张量
                features2: torch.Tensor) -> torch.Tensor:  # 第二组特征张量
        """
        计算一致性损失（自动适配维度）

        Args:
            features1: 特征1
            features2: 特征2

        Returns:
            损失值
        """
        # 确保特征维度一致：将高维特征映射到低维特征的维度
        dim1, dim2 = features1.shape[-1], features2.shape[-1]
        if dim1 != dim2:
            if dim1 > dim2:
                # features1高维 -> 映射到dim2
                features1 = self._get_dim_mapper(dim1, dim2)(features1)
            else:
                # features2高维 -> 映射到dim1
                features2 = self._get_dim_mapper(dim2, dim1)(features2)

        # 如果是MSE损失类型
        if self.loss_type == 'mse':
            # 计算两组特征之间的均方误差
            return self.loss_fn(features1, features2)

        # 如果是KL散度损失类型
        elif self.loss_type == 'kl':
            # KL散度
            # 对第一组特征应用log_softmax，并除以温度参数
            p = F.log_softmax(features1 / self.temperature, dim=-1)
            # 对第二组特征应用softmax，并除以温度参数
            q = F.softmax(features2 / self.temperature, dim=-1)
            # 计算KL散度，并乘以温度参数的平方进行缩放
            return self.loss_fn(p, q) * (self.temperature ** 2)

        # 如果是余弦相似度损失类型
        elif self.loss_type == 'cosine':
            # 余弦相似度损失
            # 创建目标张量，全为1，表示希望两组特征相似
            target = torch.ones(features1.shape[0]).to(features1.device)
            # 计算余弦嵌入损失
            return self.loss_fn(features1, features2, target)

        # 如果损失类型不匹配任何已知类型，返回0损失
        return torch.tensor(0.0).to(features1.device)


class MultiModalFusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self,
                 modalities: List[str],  # 模态名称列表
                 input_dims: Dict[str, int],  # 各模态的输入维度字典
                 hidden_dim: int = 256,  # 隐藏层维度，默认为256
                 num_classes: int = 1,  # 分类类别数，默认为1（二分类或回归）
                 feature_dims: List[int] = [64, 128, 256],  # 特征金字塔各层维度，默认为[64, 128, 256]
                 fusion_strategy: str = 'attention',  # 融合策略，默认为'attention'
                 dropout: float = 0.2,  # Dropout丢弃率，默认为0.2
                 use_consistency_loss: bool = True):  # 是否使用一致性损失，默认为True
        """
        初始化多模态融合模型

        Args:
            modalities: 模态列表
            input_dims: 各模态输入维度
            hidden_dim: 隐藏维度
            num_classes: 分类数
            feature_dims: 特征金字塔维度
            fusion_strategy: 融合策略
            dropout: Dropout率
            use_consistency_loss: 是否使用一致性损失
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()

        # 存储模态列表
        self.modalities = modalities
        # 存储是否使用一致性损失的标志
        self.use_consistency_loss = use_consistency_loss

        # 创建层次化融合网络实例
        # 层次化融合
        self.hierarchical_fusion = HierarchicalFusionNetwork(
            modalities=modalities,  # 传入模态列表
            input_dims=input_dims,  # 传入各模态输入维度
            feature_dims=feature_dims,  # 传入特征金字塔维度
            fusion_strategy=fusion_strategy,  # 传入融合策略
            dropout=dropout  # 传入Dropout率
        )

        # 创建分类头模块，用于最终的分类任务
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims[-1], hidden_dim),  # 线性层，将特征维度映射到隐藏维度
            nn.LayerNorm(hidden_dim),  # 层归一化
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim, hidden_dim // 2),  # 线性层，将隐藏维度减半
            nn.ReLU(),  # ReLU激活函数
            nn.Dropout(dropout),  # Dropout层
            nn.Linear(hidden_dim // 2, num_classes)  # 线性层，输出最终的分类结果
        )

        # 如果使用一致性损失，创建一致性损失模块
        # 一致性损失
        if use_consistency_loss:
            # 创建一致性损失实例，使用MSE损失类型
            self.consistency_loss = ConsistencyLoss(loss_type='mse')

    def forward(self,
                inputs: Dict[str, torch.Tensor],  # 各模态的输入字典
                return_features: bool = False) -> Dict[str, torch.Tensor]:  # 是否返回特征，默认为False
        """
        前向传播

        Args:
            inputs: 各模态输入
            return_features: 是否返回特征

        Returns:
            输出字典
        """
        # 通过层次化融合网络处理输入
        # 层次化融合
        fusion_outputs = self.hierarchical_fusion(inputs)

        # 对融合后的特征进行全局平均池化
        # 全局平均池化 (对序列维度)
        features = fusion_outputs['output']  # 获取融合网络的输出特征，形状为(batch, seq, features)
        features = features.mean(dim=1)  # 在序列维度上求平均，得到形状为(batch, features)的特征

        # 通过分类头进行分类
        # 分类
        logits = self.classifier(features)

        # 创建输出字典
        outputs = {
            'logits': logits,  # 分类logits
            'features': fusion_outputs['output'] if return_features else None  # 如果需要返回特征，则返回融合特征，否则返回None
        }

        # 如果使用一致性损失且模态数量大于1
        # 一致性损失
        if self.use_consistency_loss and len(self.modalities) > 1:
            # 获取各层的融合特征
            level_features = fusion_outputs['level_features']
            # 创建一个列表用于存储各相邻层之间的一致性损失
            consistency_losses = []

            # 遍历所有相邻层对
            for i in range(len(level_features) - 1):
                # 计算当前层与下一层之间的一致性损失
                loss = self.consistency_loss(level_features[i], level_features[i + 1])
                # 将损失添加到列表中
                consistency_losses.append(loss)

            # 计算所有一致性损失的平均值，并添加到输出字典中
            outputs['consistency_loss'] = sum(consistency_losses) / len(consistency_losses)

        # 返回输出字典
        return outputs


# 如果当前脚本是主程序入口
if __name__ == "__main__":
    # 测试层次化融合
    # 打印分隔线
    print("=" * 60)
    # 打印测试标题
    print("层次化融合模块测试")
    # 打印分隔线
    print("=" * 60)

    # 检测是否有可用的GPU，如果有则使用GPU，否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 打印使用的设备
    print(f"使用设备: {device}")

    # 测试特征金字塔
    # 打印测试标题
    print("\n测试特征金字塔...")
    # 创建特征金字塔实例，输入维度为64，各层特征维度为[64, 128, 256, 512]
    pyramid = FeaturePyramid(input_dim=64, feature_dims=[64, 128, 256, 512])
    # 创建随机输入张量，形状为(8, 100, 64)，即批次大小为8，序列长度为100，特征维度为64
    x = torch.randn(8, 100, 64)
    # 将输入通过特征金字塔处理
    features = pyramid(x)
    # 打印输入形状
    print(f"输入形状: {x.shape}")
    # 遍历各层特征
    for i, feat in enumerate(features):
        # 打印每一层特征的形状
        print(f"第{i}层特征形状: {feat.shape}")

    # 测试层次化融合网络
    # 打印测试标题
    print("\n测试层次化融合网络...")
    # 创建层次化融合网络实例
    hfn = HierarchicalFusionNetwork(
        modalities=['emg', 'kin'],  # 模态列表，包含emg和kin两种模态
        input_dims={'emg': 12, 'kin': 63},  # 各模态的输入维度，emg为12维，kin为63维
        feature_dims=[64, 128, 256],  # 特征金字塔各层维度
        fusion_strategy='attention'  # 使用注意力融合策略
    )

    # 创建输入字典，包含两个模态的随机数据
    inputs = {
        'emg': torch.randn(8, 100, 12),  # emg模态的输入，形状为(8, 100, 12)
        'kin': torch.randn(8, 100, 63)  # kin模态的输入，形状为(8, 100, 63)
    }

    # 将输入通过层次化融合网络处理
    outputs = hfn(inputs)
    # 打印输出特征的形状
    print(f"输出形状: {outputs['output'].shape}")
    # 打印特征层数
    print(f"特征层数: {len(outputs['level_features'])}")

    # 测试完整模型
    # 打印测试标题
    print("\n测试完整多模态融合模型...")
    # 创建多模态融合模型实例
    model = MultiModalFusionModel(
        modalities=['emg', 'kin'],  # 模态列表
        input_dims={'emg': 12, 'kin': 63},  # 各模态输入维度
        num_classes=1,  # 分类数为1
        feature_dims=[64, 128, 256]  # 特征金字塔维度
    )

    # 将输入通过模型处理，并返回特征
    outputs = model(inputs, return_features=True)
    # 打印logits的形状
    print(f"Logits形状: {outputs['logits'].shape}")
    # 打印特征的形状
    print(f"特征形状: {outputs['features'].shape}")
    # 如果输出中包含一致性损失
    if 'consistency_loss' in outputs:
        # 打印一致性损失的值
        print(f"一致性损失: {outputs['consistency_loss'].item():.4f}")

    # 打印测试完成信息
    print("\n层次化融合模块测试完成!")
