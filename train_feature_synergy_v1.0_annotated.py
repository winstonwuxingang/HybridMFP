# -*- coding: utf-8 -*-
"""
================================================================================
HybridSCIMFP - 策略3：领域知识特征（肌肉协同性）- 详细注释版
================================================================================

【项目背景】
本代码用于预测中风患者的Fugl-Meyer上肢功能评估量表（FMA-UE）评分。
基于康复医学领域知识，提取肌肉协同性特征。

【核心思想】
类比：把肌肉协同性想象成"团队协作"
- 正常人的肌肉像一个高效团队，分工明确、配合默契
- 中风患者的肌肉像一个混乱团队，分工模糊、配合失调
- 我们通过分析"团队协作模式"来评估运动功能恢复程度

【新增特征】
1. 肌肉协同性指标 - NMF分解（类似分析团队的组织结构）
2. 运动一致性指标 - 多次重复运动的变异系数（类似评估团队稳定性）
3. 代偿运动指标 - 非目标关节的运动量（类似发现有人在"越俎代庖"）

【技术栈】
- Python 3.8+
- NumPy, Pandas（数据处理）
- SciPy（信号处理）
- scikit-learn（机器学习）

【作者】DuMate AI Assistant
【日期】2026-04-16
================================================================================
"""

# ==================== 导入库 ====================
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV
import warnings

# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# 将当前目录添加到系统路径，以便导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


# ==================== 配置类 ====================
class Config:
    """
    配置类：集中管理所有配置参数
    
    【设计原则】
    将所有配置集中在一个类中，便于管理和修改。
    类比：像一个"控制面板"，所有设置一目了然。
    
    【参数说明】
    - DATA_DIR: 数据目录路径
    - OUTPUT_DIR: 输出目录路径
    - TARGET_LENGTH: DTW对齐后的目标长度（统一时间维度）
    - FMA_VALUES: FMA评分的可能取值列表
    """
    
    # 数据目录：存放原始EMG和运动学数据
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    
    # 输出目录：存放预测结果和模型报告
    OUTPUT_DIR = "./outputs/synergy_v1.0"
    
    # 目标长度：将所有样本对齐到相同的时间点数
    # 类比：把不同长度的歌曲都剪辑成同样长度
    TARGET_LENGTH = 100
    
    # FMA评分的可能取值（6个离散等级）
    # 范围：14-20，分数越高表示运动功能越好
    FMA_VALUES = [14, 16, 17, 18, 19, 20]


# ==================== DTW对齐类 ====================
class DTWAligner:
    """
    DTW对齐器：将不同长度的信号对齐到相同长度
    
    【为什么需要对齐？】
    不同受试者的运动速度不同，导致采集的信号长度不一致。
    类比：不同人唱歌速度不同，但我们需要比较同一首歌的"旋律"。
    
    【解决方法】
    使用线性插值将所有信号拉伸/压缩到相同长度。
    类比：把不同长度的橡皮筋都拉伸到100厘米。
    
    【参数说明】
    - target_length: 目标长度（默认100个时间点）
    """
    
    def __init__(self, target_length=100):
        """
        初始化DTW对齐器
        
        【参数】
        target_length: int, 目标时间点数
            - 默认值：100
            - 建议范围：50-200
            - 过小：丢失信息
            - 过大：增加计算量，但信息不增加
        """
        self.target_length = target_length
    
    def align(self, emg, kin):
        """
        对齐EMG和运动学信号
        
        【输入】
        emg: np.array, shape=(T_emg, C_emg)
            - T_emg: 原始EMG时间点数
            - C_emg: EMG通道数（通常8-16个）
        
        kin: np.array, shape=(T_kin, C_kin)
            - T_kin: 原始运动学时间点数
            - C_kin: 运动学通道数（通常63个关节角度）
        
        【输出】
        emg_aligned: np.array, shape=(target_length, C_emg)
        kin_aligned: np.array, shape=(target_length, C_kin)
        
        【处理流程】
        1. 创建目标时间轴（0到1均匀分布的100个点）
        2. 创建原始时间轴（0到1均匀分布的原始长度点）
        3. 对每个通道进行线性插值
        """
        # 获取原始信号长度
        T_emg = len(emg)   # EMG信号的时间点数
        T_kin = len(kin)   # 运动学信号的时间点数
        
        # 创建目标时间轴：[0, 0.01, 0.02, ..., 1.0]
        # 类比：把一首歌分成100个等分
        x_target = np.linspace(0, 1, self.target_length)
        
        # ===== 对齐EMG信号 =====
        # 创建原始EMG时间轴
        x_emg = np.linspace(0, 1, T_emg)
        
        # 初始化对齐后的EMG数组
        emg_aligned = np.zeros((self.target_length, emg.shape[1]))
        
        # 对每个EMG通道进行插值
        # 类比：把每条音轨都拉伸到相同长度
        for i in range(emg.shape[1]):
            # 创建插值函数：根据原始数据点，推断任意位置的值
            # kind='linear': 线性插值（相邻点之间用直线连接）
            # fill_value='extrapolate': 超出范围时外推
            f = interp1d(x_emg, emg[:, i], kind='linear', fill_value='extrapolate')
            
            # 在目标时间点上计算插值结果
            emg_aligned[:, i] = f(x_target)
        
        # ===== 对齐运动学信号 =====
        # 创建原始运动学时间轴
        x_kin = np.linspace(0, 1, T_kin)
        
        # 初始化对齐后的运动学数组
        kin_aligned = np.zeros((self.target_length, kin.shape[1]))
        
        # 对每个运动学通道进行插值
        for i in range(kin.shape[1]):
            f = interp1d(x_kin, kin[:, i], kind='linear', fill_value='extrapolate')
            kin_aligned[:, i] = f(x_target)
        
        return emg_aligned, kin_aligned


# ==================== 肌肉协同性特征提取器 ====================
class SynergyFeatureExtractor:
    """
    肌肉协同性特征提取器
    
    【核心概念】
    肌肉协同性是指中枢神经系统通过组合少数"协同模块"来控制多块肌肉的能力。
    
    【类比理解】
    想象一个管弦乐队：
    - 每个乐器 = 一块肌肉
    - 每个声部 = 一个协同模块
    - 指挥 = 中枢神经系统
    
    正常人的运动：指挥精准控制每个声部，演奏出和谐的乐章
    中风患者的运动：指挥失去部分控制，声部之间配合混乱
    
    【技术方法】
    使用非负矩阵分解（NMF）将EMG信号分解为协同模块。
    
    【参数说明】
    - n_synergies: 协同模块数量
        - 默认值：3
        - 建议范围：2-5
        - 依据：人体上肢运动通常由3-5个协同控制
    """
    
    def __init__(self, n_synergies=3):
        """
        初始化肌肉协同性特征提取器
        
        【参数】
        n_synergies: int, 协同模块数量
            - 默认值：3
            - 建议公式：n_synergies = min(3, n_channels)
            - 理论依据：人体上肢运动通常由3-5个协同控制
        """
        self.n_synergies = n_synergies

    #提取肌肉协同性特征
    def extract_synergy_features(self, emg):
        """
        提取肌肉协同性特征
        
        【输入】
        emg: np.array, shape=(T, C)
            - T: 时间点数
            - C: EMG通道数
        
        【输出】
        features: np.array, shape=(21,)
            - 21维特征向量
        
        【处理流程】
        1. 确保数据非负（NMF要求）
        2. NMF分解得到W和H矩阵
        3. 提取W矩阵特征（协同激活模式）
        4. 提取H矩阵特征（协同权重分布）
        5. 计算协同激活时间比例
        6. 计算协同间协调性
        7. 计算重构误差
        
        【类比】
        把EMG信号想象成"团队工作记录"：
        - W矩阵 = 每个时刻各团队的工作强度
        - H矩阵 = 每个团队负责的成员
        """
        features = []
        
        try:
            # ===== 步骤1：确保数据非负 =====
            # NMF要求所有元素非负
            # 取绝对值：将负值转为正值
            # 加1e-10：避免零值导致数值问题
            # 类比：把"负债"转为"资产"，并确保每个人至少有一分钱
            emg_nonneg = np.abs(emg) + 1e-10
            
            # ===== 步骤2：NMF分解 =====
            # 确定协同数量（不超过通道数）
            n_components = min(self.n_synergies, emg.shape[1])
            
            # 创建NMF模型
            # n_components: 协同模块数量
            # random_state: 随机种子，确保结果可复现
            # max_iter: 最大迭代次数，增加可提高精度
            nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
            
            # 执行NMF分解
            # 输入：emg_nonneg, shape=(T, C)
            # 输出：
            #   - W: shape=(T, K), 协同激活模式
            #   - H: shape=(K, C), 协同权重矩阵
            # 数学关系：emg ≈ W × H
            W = nmf.fit_transform(emg_nonneg)
            H = nmf.components_
            
            # ===== 步骤3：提取W矩阵特征（协同激活模式）=====
            # W矩阵：每个时刻各协同的激活强度
            # 类比：记录每个团队在每个时刻的工作强度
            
            # W的均值：各协同的平均激活强度
            # 高均值 = 这个协同"工作努力"
            features.extend(W.mean(axis=0))  # K个特征
            
            # W的标准差：各协同的激活变异性
            # 高标准差 = 这个协同"工作不稳定"
            features.extend(W.std(axis=0))   # K个特征
            
            # ===== 步骤4：提取H矩阵特征（协同权重分布）=====
            # H矩阵：每个协同对各肌肉的控制权重
            # 类比：记录每个团队负责哪些成员
            
            # H的均值：各协同控制的平均肌肉数
            # 高均值 = 这个协同"管理多块肌肉"
            features.extend(H.mean(axis=1))  # K个特征
            
            # H的标准差：各协同的肌肉选择性
            # 高标准差 = 这个协同"偏心某些肌肉"
            features.extend(H.std(axis=1))   # K个特征
            
            # ===== 步骤5：计算协同激活时间比例 =====
            # 统计每个协同在运动过程中活跃的时间比例
            # 活跃定义：激活强度 > 平均激活强度
            # 类比：统计每个团队"实际工作"的时间占比
            for i in range(n_components):
                # 计算活跃时间点数
                active_time = (W[:, i] > W[:, i].mean()).sum() / len(W)
                features.append(active_time)  # K个特征
            
            # ===== 步骤6：计算协同间协调性 =====
            # 计算不同协同之间的相关性
            # 类比：统计团队之间的配合程度
            if n_components > 1:
                # 计算W矩阵列之间的相关系数
                # corr[i,j] = 协同i和协同j的激活相关性
                corr = np.corrcoef(W.T)
                
                if not np.isnan(corr).any():
                    # 提取非对角线元素（协同间的相关性）
                    # 对角线是自己和自己的相关性，恒为1，无意义
                    off_diag = corr[np.triu_indices(n_components, k=1)]
                    
                    # 相关性均值：协同间的平均协调程度
                    # 正值 = 协同工作，负值 = 轮流工作
                    features.extend([
                        np.mean(off_diag),  # 平均协调性
                        np.std(off_diag),   # 协调性变异性
                    ])
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
            
            # ===== 步骤7：计算重构误差 =====
            # 用W和H重构EMG信号，计算与原始信号的差异
            # 类比：检查团队协作能否还原原始工作记录
            reconstruction = W @ H
            mse = np.mean((emg_nonneg - reconstruction) ** 2)
            features.append(mse)  # 1个特征
            
            # 总特征数：K*6 + 3 = 3*6 + 3 = 21维
            
        except Exception as e:
            # 如果NMF分解失败，返回默认特征
            # 避免程序崩溃
            n_default = self.n_synergies * 6 + 3
            features = [0] * n_default
        
        return np.array(features, dtype=np.float32)

    #提取运动一致性特征（多次重复运动的变异）
    def extract_movement_consistency(self, kin_repeated):
        """
        提取运动一致性特征（多次重复运动的变异）
        
        【输入】
        kin_repeated: list of np.array
            - 每个元素是一次重复运动的运动学数据
        
        【输出】
        features: np.array, shape=(5,)
            - 5维特征向量
        
        【物理意义】
        分析受试者多次重复相同运动时的稳定性。
        - 高FMA患者：运动一致性好，变异小
        - 低FMA患者：运动一致性差，变异大
        
        【类比】
        让一个人重复写同一个字10遍：
        - 正常人：10个字几乎一样
        - 中风患者：10个字大小、形状差异很大
        """
        features = []
        
        # 如果重复次数太少，无法计算一致性
        if len(kin_repeated) < 2:
            return np.array([0, 0, 0], dtype=np.float32)
        
        try:
            # ===== 步骤1：计算每次运动的统计特征 =====
            movement_features = []
            for kin in kin_repeated:
                # 计算速度（位置差分）
                vel = np.diff(kin, axis=0)
                
                # 计算速度幅值（向量长度）
                vel_mag = np.sqrt(np.sum(vel**2, axis=1))
                
                if len(vel_mag) > 0:
                    movement_features.append([
                        np.mean(vel_mag),   # 平均速度
                        np.std(vel_mag),    # 速度标准差
                        np.max(vel_mag),    # 峰值速度
                    ])
            
            # ===== 步骤2：计算变异系数 =====
            if len(movement_features) > 1:
                movement_features = np.array(movement_features)
                
                # 变异系数 = 标准差 / 均值
                # 衡量多次运动之间的相对变异程度
                # 类比：10次考试成绩的标准差/平均分
                cv = np.std(movement_features, axis=0) / (np.mean(movement_features, axis=0) + 1e-10)
                features.extend(cv)  # 3个特征
                
                # ===== 步骤3：计算运动间相似度 =====
                if len(movement_features) > 2:
                    # 计算所有运动对之间的欧氏距离
                    distances = []
                    for i in range(len(movement_features)):
                        for j in range(i+1, len(movement_features)):
                            d = np.sqrt(np.sum((movement_features[i] - movement_features[j])**2))
                            distances.append(d)
                    
                    # 平均距离：运动间的平均差异
                    features.append(np.mean(distances))
                    # 距离标准差：运动间差异的变异性
                    features.append(np.std(distances))
                else:
                    features.extend([0, 0])
            else:
                features = [0, 0, 0, 0, 0]
            
        except Exception as e:
            features = [0, 0, 0, 0, 0]
        
        return np.array(features, dtype=np.float32)


# ==================== 特征提取器 ====================
class FeatureExtractor:
    """
    特征提取器：提取EMG和运动学特征
    
    【特征类别】
    1. EMG时域特征：均值、方差、波形长度等
    2. EMG频域特征：功率谱、中位频率等
    3. EMG协方差特征：通道间相关性
    4. EMG非线性特征：峰度、偏度
    5. 肌肉协同性特征：NMF分解结果（新增）
    6. 运动学特征：位置、速度、加速度、平滑度
    
    【参数说明】
    - fs_emg: EMG采样频率（Hz），默认1000
    - fs_kin: 运动学采样频率（Hz），默认100
    """
    
    def __init__(self, fs_emg=1000, fs_kin=100):
        """
        初始化特征提取器
        
        【参数】
        fs_emg: int, EMG采样频率
            - 默认值：1000 Hz
            - 说明：每秒采集1000个样本点
        
        fs_kin: int, 运动学采样频率
            - 默认值：100 Hz
            - 说明：每秒采集100个样本点
        """
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        
        # 创建肌肉协同性特征提取器
        self.synergy_extractor = SynergyFeatureExtractor(n_synergies=3)

    #提取EMG时域特征
    def extract_emg_time_features(self, emg):
        """
        提取EMG时域特征
        
        【输入】
        emg: np.array, shape=(T, C)
            - T: 时间点数
            - C: EMG通道数
        
        【输出】
        features: np.array, shape=(C*12,)
            - 每个通道12个时域特征
        
        【特征列表】
        1. MAV (Mean Absolute Value): 平均绝对值
        2. RMS (Root Mean Square): 均方根
        3. VAR (Variance): 方差
        4. STD (Standard Deviation): 标准差
        5. MAX: 最大值
        6. MIN: 最小值
        7. RANGE: 范围（最大值-最小值）
        8. WL (Waveform Length): 波形长度
        9. Mean Diff: 平均变化率
        10. STD Diff: 变化率标准差
        11. ZC (Zero Crossing): 过零率
        12. SSC (Slope Sign Changes): 斜率符号变化
        """
        features = []
        
        # 对每个EMG通道提取特征
        for ch in range(emg.shape[1]):
            x = emg[:, ch]  # 当前通道的信号
            
            # 基础统计特征
            features.extend([
                np.mean(np.abs(x)),           # MAV: 平均绝对值
                np.sqrt(np.mean(x**2)),       # RMS: 均方根
                np.var(x),                     # VAR: 方差
                np.std(x),                     # STD: 标准差
                np.max(x),                     # MAX: 最大值
                np.min(x),                     # MIN: 最小值
                np.max(x) - np.min(x),         # RANGE: 范围
            ])
            
            # 波形特征
            diff = np.diff(x)  # 一阶差分
            features.extend([
                np.sum(np.abs(diff)) / len(diff),  # WL: 波形长度
                np.mean(np.abs(diff)),              # 平均变化率
                np.std(diff),                       # 变化率标准差
            ])
            
            # 过零率：信号穿过零点的次数
            zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
            features.append(zero_crossings / len(x))
            
            # 斜率符号变化：差分符号变化的次数
            slope_sign_changes = np.sum(np.abs(np.diff(np.sign(diff)))) / 2
            features.append(slope_sign_changes / len(diff) if len(diff) > 0 else 0)
        
        return np.array(features, dtype=np.float32)

    #　提取EMG频域特征
    def extract_emg_freq_features(self, emg):
        """
        提取EMG频域特征
        
        【输入】
        emg: np.array, shape=(T, C)
        
        【输出】
        features: np.array, shape=(C*9,)
            - 每个通道9个频域特征
        
        【特征列表】
        1-5. 各频段功率比例（低频、中低频、中频、中高频、高频）
        6. MDF (Median Frequency): 中位频率
        7. MNF (Mean Frequency): 均值频率
        8. PKF (Peak Frequency): 峰值频率
        9. SE (Spectral Entropy): 频谱熵
        """
        features = []
        
        # 定义频段（Hz）
        freq_bands = {
            'low': (0, 50),        # 低频段
            'mid_low': (50, 100),  # 中低频段
            'mid': (100, 200),     # 中频段
            'mid_high': (200, 350),# 中高频段
            'high': (350, 500)     # 高频段
        }
        
        # 对每个EMG通道提取特征
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            
            # 计算功率谱密度（Welch方法）
            # nperseg: 每段长度，影响频率分辨率
            freqs, psd = welch(x, fs=self.fs_emg, nperseg=min(256, len(x)))
            
            if len(psd) > 0 and np.sum(psd) > 0:
                total_power = np.sum(psd)
                
                # 各频段功率比例
                for band_name, (low, high) in freq_bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    band_power = np.sum(psd[band_mask]) if band_mask.sum() > 0 else 0
                    features.append(band_power / (total_power + 1e-10))
                
                # 中位频率：累积功率达到50%时的频率
                cumsum = np.cumsum(psd)
                median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
                features.append(median_freq)
                
                # 均值频率：频率的加权平均
                mean_freq = np.sum(freqs * psd) / (total_power + 1e-10)
                features.append(mean_freq)
                
                # 峰值频率：功率最大的频率
                peak_freq = freqs[np.argmax(psd)]
                features.append(peak_freq)
                
                # 频谱熵：频率分布的不确定性
                psd_norm = psd / (total_power + 1e-10)
                spec_entropy = entropy(psd_norm + 1e-10)
                features.append(spec_entropy)
            else:
                features.extend([0] * 9)
        
        return np.array(features, dtype=np.float32)

    # 提取EMG协方差特征
    def extract_emg_covariance_features(self, emg):
        """
        提取EMG协方差特征
        
        【输入】
        emg: np.array, shape=(T, C)
        
        【输出】
        features: np.array, shape=(12,)
            - 12个协方差特征
        
        【物理意义】
        分析不同EMG通道之间的相关性，反映肌肉协调性。
        - 高相关性：肌肉同步激活
        - 低相关性：肌肉独立控制
        """
        features = []
        
        # 计算协方差矩阵
        cov_matrix = np.cov(emg.T)
        
        # 对角线元素（各通道方差）
        diag = np.diag(cov_matrix)
        features.extend([
            np.mean(diag),   # 平均方差
            np.std(diag),    # 方差标准差
            np.max(diag),    # 最大方差
            np.min(diag),    # 最小方差
        ])
        
        # 非对角线元素（通道间协方差）
        n_channels = emg.shape[1]
        if n_channels > 1:
            off_diag = cov_matrix[np.triu_indices(n_channels, k=1)]
            features.extend([
                np.mean(np.abs(off_diag)),  # 平均协方差
                np.std(off_diag),           # 协方差标准差
                np.max(off_diag),           # 最大协方差
                np.min(off_diag),           # 最小协方差
            ])
            
            # 相关性矩阵
            corr_matrix = np.corrcoef(emg.T)
            if not np.isnan(corr_matrix).any():
                off_diag_corr = corr_matrix[np.triu_indices(n_channels, k=1)]
                features.extend([
                    np.mean(off_diag_corr),  # 平均相关性
                    np.std(off_diag_corr),   # 相关性标准差
                    np.max(off_diag_corr),   # 最大相关性
                    np.min(off_diag_corr),   # 最小相关性
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)

    # 提取EMG非线性特征
    def extract_emg_nonlinear_features(self, emg):
        """
        提取EMG非线性特征
        
        【输入】
        emg: np.array, shape=(T, C)
        
        【输出】
        features: np.array, shape=(C*3,)
            - 每个通道3个非线性特征
        
        【特征列表】
        1. Kurtosis: 峰度（分布的尖锐程度）
        2. Skewness: 偏度（分布的不对称程度）
        3. Complexity: 复杂度（差分标准差/原始标准差）
        """
        features = []
        
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            
            # 峰度：衡量分布的"尖峭"程度
            # 正态分布峰度为0，高峰度表示有更多极端值
            features.append(kurtosis(x))
            
            # 偏度：衡量分布的不对称性
            # 正偏度表示右侧有长尾，负偏度表示左侧有长尾
            features.append(skew(x))
            
            # 复杂度：衡量信号的复杂程度
            # 高复杂度 = 信号变化剧烈
            if len(x) > 10:
                diff = np.diff(x)
                features.append(np.std(diff) / (np.std(x) + 1e-10))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)

    # 提取所有EMG特征
    def extract_emg_features(self, emg):
        """
        提取所有EMG特征
        
        【输入】
        emg: np.array, shape=(T, C)
        
        【输出】
        features: np.array, shape=(N,)
            - N = 时域特征 + 频域特征 + 协方差特征 + 非线性特征 + 协同性特征
        """
        # 提取各类特征
        time_feat = self.extract_emg_time_features(emg)
        freq_feat = self.extract_emg_freq_features(emg)
        cov_feat = self.extract_emg_covariance_features(emg)
        nonlinear_feat = self.extract_emg_nonlinear_features(emg)
        
        # 新增：肌肉协同性特征
        synergy_feat = self.synergy_extractor.extract_synergy_features(emg)
        
        # 合并所有特征
        return np.concatenate([time_feat, freq_feat, cov_feat, nonlinear_feat, synergy_feat])

    # 提取运动学特征
    def extract_kin_features(self, kin):
        """
        提取运动学特征
        
        【输入】
        kin: np.array, shape=(T, D)
            - T: 时间点数
            - D: 运动学维度数
        
        【输出】
        features: np.array, shape=(11,)
            - 11个运动学特征
        
        【特征列表】
        1-5. 基础统计：均值、标准差、最大值、最小值、范围
        6-8. 速度特征：平均速度、速度标准差、峰值速度
        9-10. 加速度特征：平均加速度、加速度标准差
        11. 平滑度：基于jerk成本
        """
        features = []
        
        # 基础统计特征
        features.extend([
            np.mean(kin),               # 均值
            np.std(kin),                # 标准差
            np.max(kin),                # 最大值
            np.min(kin),                # 最小值
            np.max(kin) - np.min(kin),  # 范围
        ])
        
        # 速度特征
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        
        if len(vel_mag) > 0:
            features.extend([
                np.mean(vel_mag),   # 平均速度
                np.std(vel_mag),    # 速度标准差
                np.max(vel_mag),    # 峰值速度
            ])
        else:
            features.extend([0, 0, 0])
        
        # 加速度特征
        if len(vel) > 1:
            acc = np.diff(vel, axis=0)
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            
            if len(acc_mag) > 0:
                features.extend([
                    np.mean(acc_mag),  # 平均加速度
                    np.std(acc_mag),   # 加速度标准差
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # 平滑度特征（基于jerk成本）
        # jerk = 加速度的导数
        # 平滑度 = 1 / (1 + jerk成本)
        # 高平滑度 = 运动流畅
        # 低平滑度 = 运动抖动
        if len(kin) > 3:
            vel = np.gradient(kin, axis=0)
            acc = np.gradient(vel, axis=0)
            jerk = np.gradient(acc, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            
            if len(jerk_mag) > 0:
                jerk_cost = np.sum(jerk_mag**2) * 0.01  # 归一化
                smoothness = 1 / (1 + jerk_cost)
                features.append(smoothness)
            else:
                features.append(0)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)


# ==================== 受试者级特征聚合 ====================
def extract_subject_features(sample_features):
    """
    从样本级特征提取受试者级特征
    
    【输入】
    sample_features: np.array, shape=(N_samples, N_features)
        - N_samples: 该受试者的样本数
        - N_features: 样本级特征维度
    
    【输出】
    subject_feat: np.array, shape=(N_features*9,)
        - 每个原始特征聚合为9个统计量
    
    【聚合方法】
    对每个特征维度计算9个统计量：
    1. 均值
    2. 标准差
    3. 最小值
    4. 最大值
    5. 中位数
    6. 25分位数
    7. 75分位数
    8. 峰度
    9. 偏度
    
    【类比】
    把一个学生的多次考试成绩汇总为一个综合评价：
    - 平均分、标准差、最高分、最低分等
    """
    subject_feat = []
    
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        
        subject_feat.extend([
            np.mean(feat),                           # 均值
            np.std(feat),                            # 标准差
            np.min(feat),                            # 最小值
            np.max(feat),                            # 最大值
            np.median(feat),                         # 中位数
            np.percentile(feat, 25),                 # 25分位数
            np.percentile(feat, 75),                 # 75分位数
            kurtosis(feat) if len(feat) > 3 else 0,  # 峰度
            skew(feat) if len(feat) > 3 else 0,      # 偏度
        ])
    
    return np.array(subject_feat, dtype=np.float32)


# ==================== 数据加载和特征提取 ====================
def load_and_extract_features():
    """
    加载数据并提取特征
    
    【处理流程】
    1. 加载原始数据
    2. DTW对齐
    3. 提取样本级特征
    4. 聚合到受试者级
    
    【输出】
    X: np.array, shape=(N_subjects, N_features)
        - 受试者级特征矩阵
    y: np.array, shape=(N_subjects,)
        - FMA评分标签
    sids: list
        - 受试者ID列表
    """
    print("="*60)
    print("加载数据并提取特征（含肌肉协同性）...")
    print("="*60)
    
    # 加载数据
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    # 创建对齐器和特征提取器
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = FeatureExtractor(fs_emg=1000, fs_kin=100)
    
    # 存储每个受试者的样本
    subject_samples = defaultdict(list)
    subject_fma = {}
    
    print("\n特征提取中...")
    for sample in raw_samples:
        # 跳过无效样本
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue
        if sample.fma_score is None:
            continue
        
        try:
            # DTW对齐
            emg_aligned, kin_aligned = dtw_aligner.align(
                sample.emg_data, 
                sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            )
            
            # 提取特征
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
            feat = np.concatenate([emg_feat, kin_feat])
            
            # 跳过包含NaN或Inf的特征
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            
            # 存储特征
            sid = f"S{sample.subject_id}"
            subject_samples[sid].append(feat)
            subject_fma[sid] = sample.fma_score
            
        except Exception as e:
            continue
    
    # 聚合到受试者级
    X, y, sids = [], [], []
    
    print("\n聚合到受试者级别...")
    for sid, feats in subject_samples.items():
        if len(feats) > 0:
            feats_array = np.array(feats)
            subj_feat = extract_subject_features(feats_array)
            X.append(subj_feat)
            y.append(subject_fma[sid])
            sids.append(sid)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"\n受试者数量: {len(sids)}")
    print(f"受试者级特征维度: {X.shape[1]}")
    print(f"FMA分布: {dict(pd.Series(y).value_counts().sort_index())}")
    
    return X, y, sids


# ==================== 主训练函数 ====================
def train_synergy_v1():
    """
    主训练函数
    
    【处理流程】
    1. 加载数据并提取特征
    2. 数据预处理（NaN处理、标准化）
    3. Elastic Net特征选择
    4. PCA降维
    5. LOSOCV训练
    6. 评估和保存结果
    
    【模型集成】
    使用三个模型的加权集成：
    - SVR (权重0.24)
    - KNN (权重0.22)
    - GradientBoosting (权重0.54)
    """
    print("="*60)
    print("HybridSCIMFP - 策略3：领域知识特征（肌肉协同性）")
    print("="*60)
    print()
    print("新增特征:")
    print("  1. 肌肉协同性指标 (NMF分解)")
    print("  2. 协同激活模式")
    print("  3. 协同权重分布")
    print("  4. 协同间协调性")
    print()
    print("目标: MAE<1.40, Acc@1>50%")
    print("="*60)
    
    # ===== 步骤1：加载数据 =====
    X, y, sids = load_and_extract_features()
    
    # ===== 步骤2：数据预处理 =====
    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ===== 步骤3：Elastic Net特征选择 =====
    print("\n" + "="*60)
    print("Elastic Net特征选择")
    print("="*60)
    
    # ElasticNetCV参数说明：
    # cv=5: 5折交叉验证
    # l1_ratio=0.5: L1和L2正则化的混合比例
    # max_iter=10000: 最大迭代次数
    elastic = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=0.5)
    elastic.fit(X_scaled, y)
    
    # 选择非零系数对应的特征
    selected_idx = np.where(np.abs(elastic.coef_) > 1e-5)[0]
    print(f"Elastic Net选择特征数: {len(selected_idx)}")
    
    if len(selected_idx) > 0:
        X_selected = X_scaled[:, selected_idx]
    else:
        X_selected = X_scaled
    
    # ===== 步骤4：PCA降维 =====
    print("\n" + "="*60)
    print("PCA降维")
    print("="*60)
    
    if X_selected.shape[1] > 15:
        pca = PCA(n_components=min(15, X_selected.shape[1] - 1))
        X_pca = pca.fit_transform(X_selected)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA维度: {X_pca.shape[1]}, 解释方差: {explained_var:.2%}")
    else:
        X_pca = X_selected
        print(f"特征维度: {X_pca.shape[1]} (无需PCA)")
    
    # ===== 步骤5：LOSOCV训练 =====
    print("\n" + "="*60)
    print("LOSOCV训练")
    print("="*60)
    
    loo = LeaveOneOut()
    all_preds = []
    all_labels = []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test_fold = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 创建三个模型
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        # 训练模型
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权集成预测
        w_svr, w_knn, w_gb = 0.24, 0.22, 0.54
        pred = w_svr * svr.predict(X_test_fold)[0] + \
               w_knn * knn.predict(X_test_fold)[0] + \
               w_gb * gb.predict(X_test_fold)[0]
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # ===== 步骤6：计算指标 =====
    mae = np.mean(np.abs(all_preds - all_labels))
    
    # 四舍五入到最近的FMA值
    simple_round = np.array([Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds])
    acc = np.mean(simple_round == all_labels)
    acc_pm1 = np.mean(np.abs(simple_round - all_labels) <= 1)
    
    print(f"\nMAE: {mae:.2f}")
    print(f"Acc@1: {acc:.1%}")
    print(f"Acc@±1: {acc_pm1:.1%}")
    
    # 按FMA分组分析
    print("\n" + "="*60)
    print("按FMA分组分析")
    print("="*60)
    
    print(f"\n{'FMA':<8} {'数量':<8} {'预测均值':<12} {'预测标准差':<12} {'平均误差':<12}")
    print("-" * 55)
    
    for fma in sorted(set(all_labels)):
        mask = all_labels == fma
        preds = all_preds[mask]
        errors = preds - fma
        print(f"{fma:<8.0f} {mask.sum():<8} {preds.mean():<12.2f} {preds.std():<12.2f} {errors.mean():<+12.2f}")
    
    # 与之前结果对比
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    
    print(f"\n{'版本':<35} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 65)
    print(f"{'v5.8.2 (特征工程)':<35} {'0.96':<10} {'35%':<10} {'75%':<10}")
    print(f"{'策略2 (Elastic Net)':<35} {'0.72':<10} {'45%':<10} {'85%':<10}")
    print(f"{'策略3 (肌肉协同性)':<35} {mae:<10.2f} {acc:.0%}       {acc_pm1:.0%}")
    
    # ===== 步骤7：保存结果 =====
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': sids,
        'true_fma': all_labels,
        'pred_reg': all_preds,
        'pred_round': [Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds],
        'error': all_preds - all_labels,
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/synergy_v1.0_predictions.csv', index=False)
    
    summary = pd.DataFrame([{
        'version': 'synergy_v1.0',
        'strategy': '肌肉协同性特征',
        'mae': mae,
        'acc_1': acc,
        'acc_pm1': acc_pm1,
    }])
    summary.to_csv(f'{Config.OUTPUT_DIR}/synergy_v1.0_summary.csv', index=False)
    
    print(f"\n结果已保存: {Config.OUTPUT_DIR}/")
    
    return {
        'mae': mae,
        'acc': acc,
        'acc_pm1': acc_pm1,
        'preds': all_preds,
        'labels': all_labels,
    }


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    result = train_synergy_v1()
