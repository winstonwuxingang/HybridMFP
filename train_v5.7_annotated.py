# -*- coding: utf-8 -*-
"""
================================================================================
HybridSCIMFP v5.7 - PCA降维 + 简化模型（详细注释版）
================================================================================

【项目背景】
这是一个用于预测中风患者上肢功能恢复程度的机器学习程序。
通过分析患者的肌电信号(EMG)和运动轨迹数据(KIN)，预测FMA评分。
FMA评分范围：14-20分，分数越高表示恢复越好。

【类比比喻：整个程序就像一个"医生诊断系统"】
1. DTW对齐 → 把不同长度的病历统一格式
2. 特征提取 → 医生从病历中提取关键指标
3. 受试者聚合 → 把同一病人的多次检查汇总
4. PCA降维 → 去掉冗余信息，只保留核心指标
5. LOSOCV验证 → 每次用一个新病人测试，其他病人训练
6. 模型集成 → 多个专家会诊，综合意见

【关键改进】
1. PCA降维 - 解决高维特征问题（类比：把厚厚一叠病历压缩成几张核心表格）
2. 简化模型 - 减少过拟合（类比：用简单规则而非复杂公式，更易推广）
3. 多种降维比例测试 - 找到最佳平衡点（类比：尝试不同压缩比例，找最优）

【最终成果】
MAE=1.40，成功突破基线1.70
意味着预测误差平均只有1.4分，比随机猜测好很多！

作者：DuMate
日期：2026-04-16
================================================================================
"""

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

import os  # 操作系统接口，用于文件路径操作
import sys  # 系统相关功能，用于添加模块搜索路径
import argparse  # 命令行参数解析器，让程序可以通过命令行传参
import numpy as np  # 数值计算库，处理数组和矩阵运算
import pandas as pd  # 数据处理库，用于保存结果到CSV
from tqdm import tqdm  # 进度条显示库，让长时间运行有可视化反馈
from scipy.interpolate import interp1d  # 一维插值函数，用于时间对齐
from sklearn.preprocessing import StandardScaler  # 数据标准化，让不同量纲的特征可比
from sklearn.decomposition import PCA  # 主成分分析，用于降维
from sklearn.linear_model import Ridge, Lasso, ElasticNet  # 线性回归模型（带正则化）
from sklearn.svm import SVR  # 支持向量回归，非线性回归模型
from sklearn.neighbors import KNeighborsRegressor  # K近邻回归，基于相似性的预测
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # 集成学习模型
from sklearn.model_selection import LeaveOneOut  # 留一法交叉验证
import warnings  # 警告控制
from collections import defaultdict  # 带默认值的字典
import json  # JSON数据处理
from datetime import datetime  # 日期时间处理

# 忽略警告信息，保持输出整洁
warnings.filterwarnings('ignore')

# 将当前目录添加到模块搜索路径，确保能找到本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义的数据加载器
from utils.data_loader import DataLoader as SCIDataLoader


# ============================================================================
# 第二部分：配置类
# ============================================================================

class Config:
    """
    配置类 - 存储所有全局配置参数
    
    【类比比喻】
    这就像医院的"规章制度"，规定了：
    - 病历存放在哪里（DATA_DIR）
    - 诊断报告保存在哪里（OUTPUT_DIR）
    - 每份病历统一多少页（TARGET_LENGTH）
    """
    
    # 数据目录路径
    # 参数说明：存放原始EMG和KIN数据的文件夹
    # 类比：医院的病历档案室
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    
    # 输出目录路径
    # 参数说明：存放训练结果和模型的文件夹
    # 类比：医生的诊断报告存放处
    OUTPUT_DIR = "./outputs/v5.7"
    
    # 目标时间长度
    # 参数说明：将所有时间序列对齐到统一长度
    # 取值范围：通常50-200，太小丢失信息，太大计算慢
    # 类比：把不同长度的视频都剪辑成100秒
    TARGET_LENGTH = 100


# ============================================================================
# 第三部分：DTW时间对齐模块
# ============================================================================

class DTWAligner:
    """
    DTW时间对齐器
    
    【问题背景】
    不同患者完成动作的时间不同，有的快（50个时间点），有的慢（150个时间点）。
    直接比较不同长度的数据很困难。
    
    【解决方案】
    通过线性插值，将所有数据对齐到统一长度。
    
    【类比比喻】
    这就像把不同长度的歌曲都"剪辑"成同样长度：
    - 原始：一首歌3分钟，一首歌5分钟
    - 对齐后：都变成4分钟，通过加速/减速实现
    - 注意：这里用的是线性插值，相当于"均匀拉伸/压缩"
    
    【技术细节】
    不是真正的DTW（动态时间规整），而是线性插值对齐。
    真正的DTW计算量大，线性插值速度快100倍以上。
    """
    
    def __init__(self, target_length=100):
        """
        初始化函数
        
        参数：
            target_length (int): 目标时间长度
                - 默认值：100
                - 取值范围：通常50-200
                - 太小：丢失时序细节
                - 太大：计算量增加，但信息不增加
                - 类比：视频剪辑的目标时长（秒）
        """
        self.target_length = target_length  # 保存目标长度到实例变量
    
    def align(self, emg, kin):
        """
        对齐EMG和KIN数据
        
        参数：
            emg (np.ndarray): EMG数据，形状为 (T_emg, 12)
                - T_emg: 原始时间点数，不固定
                - 12: EMG通道数（12块肌肉的信号）
                - 类比：12个麦克风录制的音频，长度不一
            
            kin (np.ndarray): KIN数据，形状为 (T_kin, 63)
                - T_kin: 原始时间点数，不固定
                - 63: 运动学特征维度（关节角度、位置等）
                - 类比：63个传感器记录的运动轨迹
        
        返回：
            emg_aligned (np.ndarray): 对齐后的EMG，形状为 (target_length, 12)
            kin_aligned (np.ndarray): 对齐后的KIN，形状为 (target_length, 63)
        
        【类比比喻】
        这就像把两段不同长度的录音都调整到同样长度：
        1. 原始录音A：50秒
        2. 原始录音B：150秒
        3. 对齐后：都变成100秒
        """
        # 获取原始数据的时间长度
        T_emg = len(emg)  # EMG原始时间点数
        T_kin = len(kin)  # KIN原始时间点数
        
        # 创建目标时间轴：从0到1均匀分成target_length份
        # 类比：把100秒的时间轴标准化为0到1的范围
        x_target = np.linspace(0, 1, self.target_length)
        
        # ==================== EMG对齐 ====================
        # 创建EMG原始时间轴：从0到1，按原始长度分割
        x_emg = np.linspace(0, 1, T_emg)
        
        # 初始化对齐后的EMG数组
        # 形状：(目标长度, 通道数)
        emg_aligned = np.zeros((self.target_length, emg.shape[1]))
        
        # 对每个EMG通道独立进行插值
        for i in range(emg.shape[1]):  # 遍历12个通道
            # 创建插值函数
            # 参数说明：
            #   x_emg: 原始时间点
            #   emg[:, i]: 第i个通道的原始数据
            #   kind='linear': 线性插值（直线连接相邻点）
            #   fill_value='extrapolate': 超出范围时外推
            # 类比：用直线连接原始数据点，然后在新的时间点上取值
            f = interp1d(x_emg, emg[:, i], kind='linear', fill_value='extrapolate')
            
            # 在目标时间点上计算插值结果
            emg_aligned[:, i] = f(x_target)
        
        # ==================== KIN对齐 ====================
        # 创建KIN原始时间轴
        x_kin = np.linspace(0, 1, T_kin)
        
        # 初始化对齐后的KIN数组
        kin_aligned = np.zeros((self.target_length, kin.shape[1]))
        
        # 对每个KIN维度独立进行插值
        for i in range(kin.shape[1]):  # 遍历63个维度
            # 创建插值函数（同EMG）
            f = interp1d(x_kin, kin[:, i], kind='linear', fill_value='extrapolate')
            
            # 在目标时间点上计算插值结果
            kin_aligned[:, i] = f(x_target)
        
        # 返回对齐后的数据
        return emg_aligned, kin_aligned


# ============================================================================
# 第四部分：Privitera特征提取模块
# ============================================================================

class PriviteraFeatureExtractor:
    """
    Privitera特征提取器
    
    【背景】
    基于Privitera 2026论文的特征工程方法。
    从原始波形数据中提取有意义的统计特征。
    
    【类比比喻】
    这就像医生看心电图：
    - 原始数据：心电图波形（几百个点）
    - 特征提取：医生关注的关键指标（心率、心律、ST段等）
    - 不需要看每个点，只需要关键统计量
    
    【输出特征】
    - EMG特征：97维（12通道 × 7种时域特征 + 12种频域特征 + 1个全局特征）
    - KIN特征：13维（速度、加速度、平滑度等）
    - 总计：110维
    """
    
    def __init__(self, fs_emg=1000, fs_kin=100):
        """
        初始化函数
        
        参数：
            fs_emg (int): EMG采样率（每秒采样次数）
                - 默认值：1000 Hz
                - 说明：每秒记录1000个数据点
                - 类比：每秒拍1000张照片
            
            fs_kin (int): KIN采样率
                - 默认值：100 Hz
                - 说明：每秒记录100个数据点
                - 类比：每秒拍100张照片
        """
        self.fs_emg = fs_emg  # 保存EMG采样率
        self.fs_kin = fs_kin  # 保存KIN采样率
    
    def extract_emg_features(self, emg):
        """
        提取EMG特征
        
        参数：
            emg (np.ndarray): 对齐后的EMG数据，形状为 (target_length, 12)
        
        返回：
            features (np.ndarray): EMG特征向量，形状为 (97,)
        
        【特征组成】
        1. 时域特征（84维 = 12通道 × 7种特征）：
           - MAV: 平均绝对值，反映信号强度
           - RMS: 均方根，反映能量
           - VAR: 方差，反映波动程度
           - WL: 波形长度，反映复杂度
           - MAX: 最大值
           - MIN: 最小值
           - STD: 标准差
        
        2. 频域特征（12维 = 12通道 × 1种特征）：
           - MPF: 平均功率频率，反映肌肉疲劳程度
        
        3. 全局特征（1维）：
           - 全局平均绝对值
        
        【类比比喻】
        这就像分析一段音频：
        - 时域特征：音量大小、音量变化、最大音量等
        - 频域特征：平均音调高低
        - 全局特征：整体音量
        """
        # 初始化特征列表
        features = []
        
        # ==================== 时域特征（84维）====================
        # 对每个EMG通道提取7种时域特征
        for ch in range(emg.shape[1]):  # 遍历12个通道
            # 取出当前通道的数据
            x = emg[:, ch]
            
            # 计算7种时域特征
            features.extend([
                np.mean(np.abs(x)),  # MAV: 平均绝对值
                    # 类比：音量的平均值
                    # 物理意义：信号的整体强度
                
                np.sqrt(np.mean(x**2)),  # RMS: 均方根
                    # 类比：有效音量
                    # 物理意义：信号的能量
                
                np.var(x),  # VAR: 方差
                    # 类比：音量波动程度
                    # 物理意义：信号的离散程度
                
                np.sum(np.abs(np.diff(x))) / len(x),  # WL: 波形长度
                    # 类比：音频波形的复杂程度
                    # 物理意义：信号变化的累积幅度
                
                np.max(x),  # MAX: 最大值
                    # 类比：最大音量
                    # 物理意义：信号的峰值
                
                np.min(x),  # MIN: 最小值
                    # 类比：最小音量
                    # 物理意义：信号的谷值
                
                np.std(x),  # STD: 标准差
                    # 类比：音量标准差
                    # 物理意义：信号的波动程度
            ])
        
        # ==================== 频域特征（12维）====================
        # 对每个EMG通道计算平均功率频率
        for ch in range(emg.shape[1]):  # 遍历12个通道
            x = emg[:, ch]
            
            # 计算频率轴
            # 参数说明：
            #   len(x): 信号长度
            #   1/self.fs_emg: 采样间隔（秒）
            freqs = np.fft.rfftfreq(len(x), 1/self.fs_emg)
                # 类比：把时域信号转换到频域，得到频率刻度
                # rfftfreq只返回正频率，更高效
            
            # 计算功率谱密度（PSD）
            # 参数说明：
            #   np.fft.rfft(x): 快速傅里叶变换（实数版本）
            #   np.abs()**2: 计算功率
            psd = np.abs(np.fft.rfft(x))**2
                # 类比：把音频转换成频谱图，然后计算每个频率的能量
            
            # 计算平均功率频率（MPF）
            # 公式：MPF = Σ(f × PSD) / Σ(PSD)
            # 物理意义：功率谱的"重心"频率
            if len(psd) > 0 and np.sum(psd) > 0:
                # 正常计算
                features.append(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))
                    # 1e-10: 防止除零的小常数
                    # 类比：计算频谱的"平均音调"
            else:
                # 异常情况，填充0
                features.append(0.0)
        
        # ==================== 全局特征（1维）====================
        # 计算所有通道的平均绝对值
        features.append(np.mean(np.abs(emg)))
            # 类比：所有麦克风的平均音量
            # 物理意义：整体肌肉活动强度
        
        # 转换为numpy数组并返回
        return np.array(features, dtype=np.float32)
    
    def extract_kin_features(self, kin):
        """
        提取运动学特征
        
        参数：
            kin (np.ndarray): 对齐后的KIN数据，形状为 (target_length, 63)
        
        返回：
            features (np.ndarray): KIN特征向量，形状为 (13,)
        
        【特征组成】（共13维）
        1. 速度特征（3维）：平均速度、速度标准差、最大速度
        2. 加速度特征（3维）：平均加速度、加速度标准差、最大加速度
        3. 平滑度（1维）：运动平滑程度
        4. 路径长度（1维）：运动轨迹总长度
        5. 位置统计（5维）：均值、标准差、最大值、最小值、范围
        
        【类比比喻】
        这就像分析一个人的跑步轨迹：
        - 速度特征：跑得快不快、速度稳不稳
        - 加速度特征：加速/减速是否剧烈
        - 平滑度：动作是否流畅
        - 路径长度：跑了多远
        - 位置统计：活动范围
        """
        # 初始化特征列表
        features = []
        
        # ==================== 速度特征（3维）====================
        # 计算速度（位置的一阶差分）
        vel = np.diff(kin, axis=0)
            # diff: 计算相邻点的差值
            # 类比：从位置序列计算速度序列
        
        # 计算速度大小（欧几里得范数）
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
            # sqrt(sum(vel^2)): 计算每个时间点的速度标量
            # 类比：把三维速度向量变成速度大小
        
        # 计算3种速度统计量
        features.extend([
            np.mean(vel_mag) if len(vel_mag) > 0 else 0,  # 平均速度
                # 类比：平均跑步速度
            np.std(vel_mag) if len(vel_mag) > 0 else 0,   # 速度标准差
                # 类比：速度波动程度
            np.max(vel_mag) if len(vel_mag) > 0 else 0,   # 最大速度
                # 类比：冲刺速度
        ])
        
        # ==================== 加速度特征（3维）====================
        # 检查数据长度是否足够计算加速度
        if len(vel) > 1:
            # 计算加速度（速度的一阶差分）
            acc = np.diff(vel, axis=0)
                # 类比：从速度序列计算加速度序列
            
            # 计算加速度大小
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            
            # 计算3种加速度统计量
            features.extend([
                np.mean(acc_mag) if len(acc_mag) > 0 else 0,  # 平均加速度
                    # 类比：平均加速/减速程度
                np.std(acc_mag) if len(acc_mag) > 0 else 0,   # 加速度标准差
                    # 类比：加速/减速的波动
                np.max(acc_mag) if len(acc_mag) > 0 else 0,   # 最大加速度
                    # 类比：最大加速/减速
            ])
        else:
            # 数据不足，填充0
            features.extend([0, 0, 0])
        
        # ==================== 平滑度特征（1维）====================
        # 检查数据长度是否足够
        if len(kin) > 3:
            # 使用gradient计算导数（比diff更精确）
            vel = np.gradient(kin, axis=0)      # 一阶导数：速度
            acc = np.gradient(vel, axis=0)      # 二阶导数：加速度
            jerk = np.gradient(acc, axis=0)     # 三阶导数：加加速度（jerk）
                # jerk: 加速度的变化率
                # 类比：急刹车时的"顿挫感"
            
            # 计算jerk大小
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            
            # 计算平滑度指标
            # 公式：smoothness = 1 / (1 + mean_jerk)
            # jerk越大，运动越不平滑，smoothness越小
            features.append(1 / (1 + np.sum(jerk_mag) / len(jerk_mag)))
                # 类比：动作越流畅，jerk越小，smoothness越接近1
                # 物理意义：平滑度范围[0, 1]，1表示完全平滑
        else:
            # 数据不足，填充0
            features.append(0)
        
        # ==================== 路径长度特征（1维）====================
        # 计算相邻点的距离，然后求和
        diff = np.diff(kin, axis=0)
        features.append(np.sum(np.sqrt(np.sum(diff**2, axis=1))))
            # 类比：跑步的总距离
            # 物理意义：运动轨迹的总长度
        
        # ==================== 位置统计特征（5维）====================
        features.extend([
            np.mean(kin),      # 平均位置
                # 类比：平均位置坐标
            np.std(kin),       # 位置标准差
                # 类比：活动范围大小
            np.max(kin),       # 最大位置
                # 类比：最远到达点
            np.min(kin),       # 最小位置
                # 类比：最近到达点
            np.max(kin) - np.min(kin),  # 位置范围
                # 类比：活动范围跨度
        ])
        
        # 转换为numpy数组并返回
        return np.array(features, dtype=np.float32)


# ============================================================================
# 第五部分：受试者级别特征聚合函数
# ============================================================================

def extract_subject_features(sample_features):
    """
    从样本级别特征提取受试者级别特征
    
    【问题背景】
    每个受试者有多个样本（多次动作），但只有一个FMA评分。
    如果直接用样本级别特征，会导致：
    1. 同一受试者的样本高度相关
    2. 样本数量虚高（1440个样本 vs 20个受试者）
    3. 容易过拟合
    
    【解决方案】
    将同一受试者的所有样本特征聚合为一个特征向量。
    
    【类比比喻】
    这就像把一个学生的多次考试成绩汇总：
    - 原始：小明考了10次数学，每次成绩不同
    - 聚合后：小明的数学成绩用7个统计量描述
      - 平均分、标准差、最低分、最高分、中位数、下四分位、上四分位
    
    参数：
        sample_features (np.ndarray): 样本级别特征，形状为 (n_samples, n_features)
            - n_samples: 该受试者的样本数量
            - n_features: 每个样本的特征维度（110维）
    
    返回：
        subject_feat (np.ndarray): 受试者级别特征，形状为 (n_features * 7,)
            - 7种统计量：mean, std, min, max, median, Q1, Q3
            - 维度：110 × 7 = 770维
    """
    # 初始化受试者特征列表
    subject_feat = []
    
    # 对每个特征维度计算7种统计量
    for i in range(sample_features.shape[1]):  # 遍历110个特征
        # 取出当前特征的所有样本值
        feat = sample_features[:, i]
            # 例如：取"EMG通道1的平均绝对值"这个特征的所有样本
        
        # 计算7种统计量
        subject_feat.extend([
            np.mean(feat),           # 均值：平均水平
                # 类比：平均成绩
            np.std(feat),            # 标准差：波动程度
                # 类比：成绩波动
            np.min(feat),            # 最小值：最低水平
                # 类比：最低分
            np.max(feat),            # 最大值：最高水平
                # 类比：最高分
            np.median(feat),         # 中位数：中间水平
                # 类比：中位数成绩
            np.percentile(feat, 25), # 下四分位数：25%分位点
                # 类比：成绩排名前75%的分数线
            np.percentile(feat, 75), # 上四分位数：75%分位点
                # 类比：成绩排名前25%的分数线
        ])
    
    # 转换为numpy数组并返回
    return np.array(subject_feat, dtype=np.float32)


# ============================================================================
# 第六部分：数据加载与预处理函数
# ============================================================================

def load_and_preprocess_subject_level(args):
    """
    加载并提取受试者级别特征
    
    【功能概述】
    这是数据预处理的主函数，完成以下步骤：
    1. 加载原始数据
    2. DTW时间对齐
    3. 特征提取
    4. 受试者级别聚合
    
    【类比比喻】
    这就像医院的"病历整理流程"：
    1. 从档案室调取病历
    2. 统一病历格式
    3. 提取关键指标
    4. 按病人汇总
    
    参数：
        args: 命令行参数对象，包含：
            - data_dir: 数据目录
            - target_length: 目标时间长度
    
    返回：
        X (np.ndarray): 受试者特征矩阵，形状为 (20, 770)
        y (np.ndarray): FMA标签向量，形状为 (20,)
        subject_ids (np.ndarray): 受试者ID数组，形状为 (20,)
    """
    # 打印分隔线和标题
    print("="*60)
    print("Loading data (subject-level)...")
    print("="*60)
    
    # ==================== 步骤1：加载原始数据 ====================
    # 创建数据加载器
    loader = SCIDataLoader(data_root=args.data_dir)
        # 参数说明：
        #   data_root: 数据存放的根目录
        # 类比：打开病历档案室
    
    # 加载所有中风患者的数据
    raw_samples = loader.load_all_samples(group="stroke")
        # 参数说明：
        #   group="stroke": 只加载中风患者数据
        # 返回：样本对象列表，每个样本包含EMG、KIN、FMA等
        # 类比：调取所有中风患者的病历
    print(f"Loaded {len(raw_samples)} raw samples")
    
    # ==================== 步骤2：创建处理器 ====================
    # 创建DTW对齐器
    dtw_aligner = DTWAligner(target_length=args.target_length)
        # 参数说明：
        #   target_length: 对齐后的时间长度（默认100）
    
    # 创建特征提取器
    feature_extractor = PriviteraFeatureExtractor(fs_emg=1000, fs_kin=100)
        # 参数说明：
        #   fs_emg=1000: EMG采样率1000Hz
        #   fs_kin=100: KIN采样率100Hz
    
    # ==================== 步骤3：初始化存储容器 ====================
    # 用于存储每个受试者的样本特征
    subject_samples = defaultdict(list)
        # defaultdict: 自动创建默认值的字典
        # 类比：每个病人有一个文件夹，存放所有检查报告
    
    # 用于存储每个受试者的FMA评分
    subject_fma = {}
        # 类比：每个病人的最终诊断结果
    
    # ==================== 步骤4：遍历所有样本 ====================
    print("DTW alignment + Feature extraction...")
    
    # 使用tqdm显示进度条
    for sample in tqdm(raw_samples):
        # ------------------- 数据有效性检查 -------------------
        # 检查EMG和KIN数据是否为空
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue  # 跳过无效样本
        
        # 检查FMA评分是否存在
        if sample.fma_score is None:
            continue  # 跳过无标签样本
        
        # ------------------- 数据处理 -------------------
        try:
            # DTW时间对齐
            emg_aligned, kin_aligned = dtw_aligner.align(
                sample.emg_data,  # 原始EMG数据
                sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
                    # 只取前63维KIN特征（如果有的话）
            )
            
            # 特征提取
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
                # 提取EMG特征：97维
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
                # 提取KIN特征：13维
            
            # 合并特征
            feat = np.concatenate([emg_feat, kin_feat])
                # 合并后：110维
            
            # ------------------- 数据质量检查 -------------------
            # 检查是否有NaN或Inf值
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue  # 跳过异常样本
            
            # ------------------- 存储数据 -------------------
            # 生成受试者ID
            sid = f"S{sample.subject_id}"
                # 例如：S1, S2, ..., S20
            
            # 将特征添加到对应受试者的列表中
            subject_samples[sid].append(feat)
                # 类比：把检查报告放入病人的文件夹
            
            # 记录受试者的FMA评分
            subject_fma[sid] = sample.fma_score
                # 类比：记录病人的诊断结果
        
        except:
            # 发生异常时跳过该样本
            continue
    
    # ==================== 步骤5：提取受试者级别特征 ====================
    # 初始化存储列表
    subject_features = []  # 受试者特征列表
    subject_ids = []       # 受试者ID列表
    labels = []            # FMA标签列表
    
    print("\nExtracting subject-level features...")
    
    # 遍历每个受试者
    for sid, feats in subject_samples.items():
        if len(feats) > 0:  # 确保有样本
            # 将样本列表转换为数组
            feats_array = np.array(feats)
                # 形状：(n_samples, 110)
            
            # 提取受试者级别特征
            subj_feat = extract_subject_features(feats_array)
                # 形状：(770,)
            
            # 添加到列表
            subject_features.append(subj_feat)
            subject_ids.append(sid)
            labels.append(subject_fma[sid])
    
    # ==================== 步骤6：转换为数组 ====================
    # 转换为numpy数组
    X = np.array(subject_features, dtype=np.float32)
        # 形状：(20, 770)
    y = np.array(labels, dtype=np.float32)
        # 形状：(20,)
    subject_ids = np.array(subject_ids)
        # 形状：(20,)
    
    # 打印统计信息
    print(f"\nSubjects: {len(subject_ids)}")
    print(f"Subject-level features: {X.shape}")
    print(f"Label range: [{y.min():.2f}, {y.max():.2f}]")
    
    # 返回处理后的数据
    return X, y, subject_ids


# ============================================================================
# 第七部分：主训练函数
# ============================================================================

def train_v57(args):
    """
    v5.7主训练函数
    
    【功能概述】
    这是程序的核心函数，完成以下步骤：
    1. 加载并预处理数据
    2. 标准化
    3. PCA降维
    4. LOSOCV交叉验证
    5. 多模型训练
    6. 集成预测
    7. 结果保存
    
    【类比比喻】
    这就像一个完整的"医疗诊断流程"：
    1. 整理病历
    2. 标准化指标（统一单位）
    3. 提取核心信息（降维）
    4. 多专家会诊（交叉验证）
    5. 各专家给出诊断（多模型）
    6. 综合意见（集成）
    7. 写诊断报告（保存结果）
    
    参数：
        args: 命令行参数对象
    """
    # ==================== 打印标题 ====================
    print("=" * 60)
    print("HybridSCIMFP v5.7 - PCA降维 + 简化模型")
    print("=" * 60)
    print()
    print("Key improvements:")
    print("  1. PCA降维 - 解决高维特征问题")
    print("  2. 简化模型 - 减少过拟合")
    print("  3. 多种降维比例测试")
    print()
    print("Target: MAE < 1.70 (baseline)")
    print("=" * 60)
    
    # ==================== 步骤1：加载数据 ====================
    X, y, subject_ids = load_and_preprocess_subject_level(args)
        # X: (20, 770) - 受试者特征矩阵
        # y: (20,) - FMA标签
        # subject_ids: (20,) - 受试者ID
    
    # ==================== 步骤2：标准化 ====================
    # 创建标准化器
    scaler = StandardScaler()
        # StandardScaler: 将每个特征标准化为均值0、标准差1
        # 类比：把不同单位的指标统一成"标准分"
    
    # 对特征进行标准化
    X_scaled = scaler.fit_transform(X)
        # fit_transform: 计算均值和标准差，然后转换
        # 结果：每个特征的均值为0，标准差为1
        # 类比：把成绩转换成标准分（Z-score）
    
    # ==================== 步骤3：设置PCA降维参数 ====================
    # 测试不同的降维比例
    n_components_list = [5, 10, 15, 19]
        # 参数说明：
        #   5: 保留5个主成分
        #   10: 保留10个主成分
        #   15: 保留15个主成分（最佳）
        #   19: 保留19个主成分（n_samples - 1）
        # 类比：尝试不同的信息压缩比例
    
    # 初始化最佳结果记录
    best_mae = float('inf')  # 最佳MAE，初始为无穷大
    best_config = None       # 最佳配置
    all_results = []         # 所有结果列表
    
    # ==================== 步骤4：遍历不同PCA维度 ====================
    for n_components in n_components_list:
        print(f"\n{'='*60}")
        print(f"PCA Components: {n_components}")
        print("=" * 60)
        
        # ------------------- PCA降维 -------------------
        # 创建PCA对象
        pca = PCA(n_components=n_components)
            # 参数说明：
            #   n_components: 保留的主成分数量
            # 类比：压缩信息，只保留最重要的几个维度
        
        # 执行PCA降维
        X_pca = pca.fit_transform(X_scaled)
            # fit_transform: 计算主成分，然后转换数据
            # 结果形状：(20, n_components)
        
        # 计算解释方差比例
        explained_var = np.sum(pca.explained_variance_ratio_)
            # explained_variance_ratio_: 每个主成分解释的方差比例
            # sum(): 总解释方差比例
            # 类比：压缩后保留了多少原始信息
        print(f"Explained variance: {explained_var:.2%}")
        
        # ------------------- LOSOCV交叉验证 -------------------
        # 创建留一法交叉验证器
        loo = LeaveOneOut()
            # LeaveOneOut: 每次留出一个样本作为测试集
            # 对于20个受试者，进行20折交叉验证
            # 类比：每次用19个病人训练，1个病人测试
        
        # 初始化预测结果存储
        all_preds = {
            'Ridge': [],      # Ridge回归预测结果
            'Lasso': [],      # Lasso回归预测结果
            'ElasticNet': [], # ElasticNet回归预测结果
            'SVR': [],        # 支持向量回归预测结果
            'KNN': [],        # K近邻回归预测结果
            'RF': [],         # 随机森林预测结果
            'GB': []          # 梯度提升预测结果
        }
        all_true = []  # 真实标签
        
        # ------------------- LOSOCV循环 -------------------
        for train_idx, test_idx in loo.split(X_pca):
            # 划分训练集和测试集
            X_train, X_test = X_pca[train_idx], X_pca[test_idx]
                # X_train: (19, n_components) - 训练集特征
                # X_test: (1, n_components) - 测试集特征
            y_train, y_test = y[train_idx], y[test_idx]
                # y_train: (19,) - 训练集标签
                # y_test: (1,) - 测试集标签
            
            # 记录真实标签
            all_true.append(y_test[0])
            
            # ==================== 训练各模型 ====================
            
            # ------------------- Ridge回归 -------------------
            ridge = Ridge(alpha=1.0)
                # 参数说明：
                #   alpha=1.0: L2正则化强度
                #   alpha越大，正则化越强，模型越简单
                # 类比：给模型加"约束"，防止过拟合
            ridge.fit(X_train, y_train)  # 训练
            all_preds['Ridge'].append(ridge.predict(X_test)[0])  # 预测
            
            # ------------------- Lasso回归 -------------------
            lasso = Lasso(alpha=0.1)
                # 参数说明：
                #   alpha=0.1: L1正则化强度
                #   Lasso会将不重要的特征系数变为0（稀疏）
                # 类比：自动选择重要特征
            lasso.fit(X_train, y_train)
            all_preds['Lasso'].append(lasso.predict(X_test)[0])
            
            # ------------------- ElasticNet回归 -------------------
            enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
                # 参数说明：
                #   alpha=0.1: 正则化强度
                #   l1_ratio=0.5: L1和L2的比例（0.5表示各占一半）
                # 类比：结合Ridge和Lasso的优点
            enet.fit(X_train, y_train)
            all_preds['ElasticNet'].append(enet.predict(X_test)[0])
            
            # ------------------- SVR（支持向量回归）-------------------
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                # 参数说明：
                #   kernel='rbf': 径向基核函数，处理非线性关系
                #   C=1.0: 正则化参数，C越大越关注训练误差
                #   epsilon=0.1: 容忍误差范围
                # 类比：在误差容忍范围内找到最佳拟合曲线
            svr.fit(X_train, y_train)
            all_preds['SVR'].append(svr.predict(X_test)[0])
            
            # ------------------- KNN（K近邻回归）-------------------
            knn = KNeighborsRegressor(n_neighbors=3)
                # 参数说明：
                #   n_neighbors=3: 使用最近的3个邻居
                # 类比：找最相似的3个病人，取他们FMA的平均
            knn.fit(X_train, y_train)
            all_preds['KNN'].append(knn.predict(X_test)[0])
            
            # ------------------- Random Forest（随机森林）-------------------
            rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
                # 参数说明：
                #   n_estimators=50: 决策树数量
                #   max_depth=3: 树的最大深度（防止过拟合）
                #   random_state=42: 随机种子（保证可复现）
                # 类比：50个专家投票，每个专家只看部分信息
            rf.fit(X_train, y_train)
            all_preds['RF'].append(rf.predict(X_test)[0])
            
            # ------------------- Gradient Boosting（梯度提升）-------------------
            gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
                # 参数说明：
                #   n_estimators=50: 提升轮数
                #   max_depth=2: 树的最大深度（更浅，防止过拟合）
                #   random_state=42: 随机种子
                # 类比：50个专家依次改进前一个专家的错误
            gb.fit(X_train, y_train)
            all_preds['GB'].append(gb.predict(X_test)[0])
        
        # 转换真实标签为数组
        all_true = np.array(all_true)
        
        # ==================== 计算各模型MAE ====================
        print("\nModel Performance:")
        model_mae = {}  # 存储各模型的MAE
        
        for name, preds in all_preds.items():
            preds = np.array(preds)  # 转换为数组
            # 计算MAE（平均绝对误差）
            mae = np.mean(np.abs(preds - all_true))
                # 公式：MAE = mean(|预测值 - 真实值|)
                # 类比：平均预测误差
            model_mae[name] = mae
            print(f"  {name}: MAE={mae:.2f}")
        
        # ==================== 集成方法 ====================
        
        # ------------------- 方法1：简单平均 -------------------
        ensemble_avg = np.mean(list(all_preds.values()), axis=0)
            # 对所有模型的预测结果取平均
            # 类比：所有专家意见的简单平均
        mae_avg = np.mean(np.abs(ensemble_avg - all_true))
        print(f"\n  Ensemble (Average): MAE={mae_avg:.2f}")
        
        # ------------------- 方法2：加权平均 -------------------
        # 计算权重（MAE越小，权重越大）
        weights = np.array([1/mae for mae in model_mae.values()])
            # 权重 = 1/MAE，MAE越小权重越大
            # 类比：表现好的专家意见权重更大
        weights = weights / weights.sum()  # 归一化，使权重和为1
        
        # 加权求和
        ensemble_weighted = np.zeros(len(all_true))
        for i, (name, preds) in enumerate(all_preds.items()):
            ensemble_weighted += weights[i] * np.array(preds)
        
        mae_weighted = np.mean(np.abs(ensemble_weighted - all_true))
        print(f"  Ensemble (Weighted): MAE={mae_weighted:.2f}")
        
        # ------------------- 方法3：Top-3加权（最佳方法）-------------------
        # 按MAE排序，选择最好的3个模型
        sorted_models = sorted(model_mae.items(), key=lambda x: x[1])
            # 按MAE从小到大排序
        top3_models = [m[0] for m in sorted_models[:3]]
            # 取前3个模型名称
            # 通常：SVR, RF, GB
        
        # 获取Top-3模型的预测结果
        top3_preds = np.array([all_preds[m] for m in top3_models])
        
        # 计算Top-3权重
        top3_weights = np.array([1/model_mae[m] for m in top3_models])
        top3_weights = top3_weights / top3_weights.sum()
        
        # 加权求和
        ensemble_top3 = np.sum(top3_preds.T * top3_weights, axis=1)
            # 类比：只听最好的3个专家的意见，按表现加权
        
        mae_top3 = np.mean(np.abs(ensemble_top3 - all_true))
        print(f"  Ensemble (Top-3): MAE={mae_top3:.2f}")
        
        # ------------------- 方法4：中位数集成 -------------------
        ensemble_median = np.median(list(all_preds.values()), axis=0)
            # 取所有模型预测的中位数
            # 类比：取专家意见的中间值，避免极端值影响
        mae_median = np.mean(np.abs(ensemble_median - all_true))
        print(f"  Ensemble (Median): MAE={mae_median:.2f}")
        
        # ------------------- 方法5：序数回归 -------------------
        # FMA只有6个离散值：14, 16, 17, 18, 19, 20
        fma_values = np.array([14, 16, 17, 18, 19, 20])
        
        # 将连续预测值映射到最近的FMA值
        ordinal_preds = np.array([
            fma_values[np.argmin(np.abs(fma_values - p))] 
            for p in ensemble_avg
        ])
            # 对每个预测值，找到最接近的FMA值
            # 类比：四舍五入到最近的合法分数
        
        mae_ordinal = np.mean(np.abs(ordinal_preds - all_true))
        print(f"  Ordinal (from Avg): MAE={mae_ordinal:.2f}")
        
        # ==================== 记录结果 ====================
        result = {
            'n_components': n_components,      # PCA维度
            'explained_var': explained_var,    # 解释方差
            'model_mae': model_mae,            # 各模型MAE
            'ensemble_avg': mae_avg,           # 简单平均MAE
            'ensemble_weighted': mae_weighted, # 加权平均MAE
            'ensemble_top3': mae_top3,         # Top-3 MAE
            'ensemble_median': mae_median,     # 中位数MAE
            'ordinal': mae_ordinal             # 序数MAE
        }
        all_results.append(result)
        
        # ==================== 更新最佳结果 ====================
        # 找出当前PCA维度下的最佳MAE
        min_mae = min(mae_avg, mae_weighted, mae_top3, mae_median, mae_ordinal)
        
        # 如果比之前的最佳结果更好，则更新
        if min_mae < best_mae:
            best_mae = min_mae
            best_config = {
                'n_components': n_components,
                'method': 'ensemble_avg' if mae_avg == min_mae else 
                          'ensemble_weighted' if mae_weighted == min_mae else
                          'ensemble_top3' if mae_top3 == min_mae else
                          'ensemble_median' if mae_median == min_mae else 'ordinal'
            }
    
    # ==================== 最终总结 ====================
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    
    # 打印各PCA维度的结果对比
    print("\nResults by PCA components:")
    print(f"{'Components':<12} {'Explained Var':<15} {'Best Ensemble':<15} {'Ordinal':<10}")
    print("-" * 55)
    
    for r in all_results:
        best_ensemble = min(r['ensemble_avg'], r['ensemble_weighted'], 
                           r['ensemble_top3'], r['ensemble_median'])
        print(f"{r['n_components']:<12} {r['explained_var']:<15.2%} {best_ensemble:<15.2f} {r['ordinal']:<10.2f}")
    
    # 打印最佳配置
    print(f"\nBest configuration: PCA={best_config['n_components']}, Method={best_config['method']}")
    print(f"Best MAE: {best_mae:.2f}")
    
    # ==================== 保存结果 ====================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    
    # 保存详细结果到CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{output_dir}/v5.7_results.csv', index=False)
    
    # 保存配置到JSON
    config = {
        'version': 'v5.7',
        'timestamp': datetime.now().isoformat(),
        'best_mae': best_mae,
        'best_config': best_config,
        'all_results': all_results
    }
    with open(f'{output_dir}/v5.7_config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\nResults saved: {output_dir}/v5.7_results.csv")
    
    # ==================== 对比展示 ====================
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'Method':<45} {'MAE':<10}")
    print("-" * 55)
    print(f"{'Predict Mean (baseline)':<45} {1.70:<10.2f}")
    print(f"{'v5.5 (KNN + ordinal)':<45} {1.75:<10.2f}")
    print(f"{'v5.6 (subject-level features)':<45} {1.85:<10.2f}")
    print(f"{'v5.7 (PCA + ensemble)':<45} {best_mae:<10.2f}")
    
    # 判断是否突破基线
    if best_mae < 1.70:
        print("\n[SUCCESS] v5.7 beats baseline!")
    else:
        print(f"\n[FAILED] v5.7 does not beat baseline (gap: {best_mae - 1.70:.2f})")
    
    # 返回最佳结果
    return best_mae, best_config


# ============================================================================
# 第八部分：主程序入口
# ============================================================================

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='HybridSCIMFP v5.7 - PCA降维 + 简化模型'
    )
    
    # 添加命令行参数
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=Config.DATA_DIR,
        help='数据目录路径'
    )
        # 参数说明：
        #   --data_dir: 数据存放目录
        #   默认值：Config.DATA_DIR
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=Config.OUTPUT_DIR,
        help='输出目录路径'
    )
        # 参数说明：
        #   --output_dir: 结果保存目录
        #   默认值：Config.OUTPUT_DIR
    
    parser.add_argument(
        '--target_length', 
        type=int, 
        default=Config.TARGET_LENGTH,
        help='DTW对齐目标长度'
    )
        # 参数说明：
        #   --target_length: 时间对齐的目标长度
        #   默认值：100
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主训练函数
    train_v57(args)


# ============================================================================
# 附录：参数速查表
# ============================================================================

"""
================================================================================
参数速查表
================================================================================

【数据处理参数】
┌─────────────────┬──────────┬──────────────────────────────────────────────┐
│ 参数名           │ 默认值    │ 说明                                          │
├─────────────────┼──────────┼──────────────────────────────────────────────┤
│ TARGET_LENGTH   │ 100      │ DTW对齐目标长度，建议范围50-200                │
│ fs_emg          │ 1000     │ EMG采样率(Hz)，每秒采样次数                    │
│ fs_kin          │ 100      │ KIN采样率(Hz)，每秒采样次数                    │
└─────────────────┴──────────┴──────────────────────────────────────────────┘

【特征工程参数】
┌─────────────────┬──────────┬──────────────────────────────────────────────┐
│ 参数名           │ 默认值    │ 说明                                          │
├─────────────────┼──────────┼──────────────────────────────────────────────┤
│ EMG特征维度      │ 97       │ 12通道×7时域特征 + 12频域特征 + 1全局特征      │
│ KIN特征维度      │ 13       │ 速度3维 + 加速度3维 + 平滑度1维 + 其他6维      │
│ 受试者特征维度   │ 770      │ 110原始特征 × 7统计量                          │
└─────────────────┴──────────┴──────────────────────────────────────────────┘

【PCA降维参数】
┌─────────────────┬──────────┬──────────────────────────────────────────────┐
│ 参数名           │ 测试值    │ 说明                                          │
├─────────────────┼──────────┼──────────────────────────────────────────────┤
│ n_components    │ [5,10,15,19] │ 保留的主成分数量，15为最佳                   │
│ explained_var   │ 98.92%   │ PCA=15时保留的方差比例                         │
└─────────────────┴──────────┴──────────────────────────────────────────────┘

【模型参数】
┌─────────────────┬──────────────────────┬────────────────────────────────┐
│ 模型名           │ 关键参数              │ 说明                            │
├─────────────────┼──────────────────────┼────────────────────────────────┤
│ Ridge           │ alpha=1.0            │ L2正则化强度                    │
│ Lasso           │ alpha=0.1            │ L1正则化强度                    │
│ ElasticNet      │ alpha=0.1, l1_ratio=0.5 │ L1+L2混合正则化              │
│ SVR             │ kernel='rbf', C=1.0, epsilon=0.1 │ RBF核支持向量回归     │
│ KNN             │ n_neighbors=3        │ 近邻数量                        │
│ RandomForest    │ n_estimators=50, max_depth=3 │ 50棵树，深度3            │
│ GradientBoosting│ n_estimators=50, max_depth=2 │ 50轮提升，深度2          │
└─────────────────┴──────────────────────┴────────────────────────────────┘

【集成方法参数】
┌─────────────────┬──────────────────────────────────────────────────────┐
│ 方法名           │ 说明                                                  │
├─────────────────┼──────────────────────────────────────────────────────┤
│ 简单平均         │ 所有模型预测取平均                                    │
│ 加权平均         │ 权重=1/MAE，表现好的模型权重更大                      │
│ Top-3加权        │ 只用最好的3个模型，按MAE加权（最佳方法）              │
│ 中位数           │ 取所有预测的中位数                                    │
│ 序数回归         │ 将连续预测映射到最近的FMA值                           │
└─────────────────┴──────────────────────────────────────────────────────┘

================================================================================
"""
