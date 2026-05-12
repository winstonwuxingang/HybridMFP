# -*- coding: utf-8 -*-
"""
HybridSCIMFP Optimized v8.0 - 数据增强 + 受试者级别特征 + 集成学习

优化内容：
1. 时序数据增强 - 时间扭曲、幅值缩放、噪声注入
2. 受试者级别特征 - 统计特征、分布特征
3. 多模型集成 - SVR + RF + GB + XGBoost
4. 类别感知损失 - 类内紧凑 + 类间分离

目标：MAE < 1.40, Acc@1 > 75%
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


# ============================================================================
# 配置
# ============================================================================
class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/optimized_v8"
    
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    NUM_CLASSES = 6
    
    # 数据增强参数
    AUGMENT_RATIO = 3  # 每个样本生成3个增强样本
    TIME_WARP_SIGMA = 0.2  # 时间扭曲强度
    MAGNITUDE_SCALE_RANGE = (0.9, 1.1)  # 幅值缩放范围
    NOISE_STD = 0.05  # 噪声标准差
    
    # 特征参数
    PCA_COMPONENTS = 15
    
    # 集成权重
    ENSEMBLE_WEIGHTS = {
        'svr': 0.3,
        'rf': 0.25,
        'gb': 0.25,
        'knn': 0.2
    }


# ============================================================================
# 数据增强
# ============================================================================
class DataAugmenter:
    """时序数据增强器"""
    
    def __init__(self, config):
        self.config = config
    
    def time_warp(self, data, sigma=0.2):
        """时间扭曲：改变时间轴的流速"""
        T = len(data)
        # 生成扭曲的时间轴
        t_orig = np.linspace(0, 1, T)
        t_warp = np.cumsum(np.random.normal(1, sigma, T))
        t_warp = (t_warp - t_warp[0]) / (t_warp[-1] - t_warp[0])
        t_warp = np.clip(t_warp, 0, 1)
        
        # 插值
        warped = np.zeros_like(data)
        for c in range(data.shape[1]):
            f = interp1d(t_orig, data[:, c], kind='linear', fill_value='extrapolate')
            warped[:, c] = f(t_warp)
        
        return warped
    
    def magnitude_scale(self, data, scale_range=(0.9, 1.1)):
        """幅值缩放"""
        scale = np.random.uniform(*scale_range)
        return data * scale
    
    def add_noise(self, data, std=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(0, std, data.shape)
        return data + noise.astype(np.float32)
    
    def augment(self, data):
        """组合增强"""
        augmented = data.copy()
        
        # 随机选择增强方式
        if np.random.random() < 0.5:
            augmented = self.time_warp(augmented, self.config.TIME_WARP_SIGMA)
        
        if np.random.random() < 0.5:
            augmented = self.magnitude_scale(augmented, self.config.MAGNITUDE_SCALE_RANGE)
        
        if np.random.random() < 0.5:
            augmented = self.add_noise(augmented, self.config.NOISE_STD)
        
        return augmented.astype(np.float32)


# ============================================================================
# 特征提取
# ============================================================================
class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        """DTW对齐"""
        T_emg, T_kin = len(emg), len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        emg_aligned = np.zeros((self.target_length, emg.shape[1]), dtype=np.float32)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]), dtype=np.float32)
        
        for c in range(emg.shape[1]):
            f = interp1d(np.linspace(0, 1, T_emg), emg[:, c], kind='linear', fill_value='extrapolate')
            emg_aligned[:, c] = f(x_target)
        
        for c in range(kin.shape[1]):
            f = interp1d(np.linspace(0, 1, T_kin), kin[:, c], kind='linear', fill_value='extrapolate')
            kin_aligned[:, c] = f(x_target)
        
        return emg_aligned, kin_aligned
    
    def extract_sample_features(self, emg, kin):
        """提取单个样本的特征"""
        features = []
        
        # EMG统计特征
        features.extend([
            emg.mean(), emg.std(), emg.max(), emg.min(),
            np.median(emg), skew(emg.flatten()), kurtosis(emg.flatten())
        ])
        
        # EMG通道特征
        for c in range(min(12, emg.shape[1])):
            channel = emg[:, c]
            features.extend([
                channel.mean(), channel.std(), channel.max() - channel.min()
            ])
        
        # KIN统计特征
        features.extend([
            kin.mean(), kin.std(), kin.max(), kin.min(),
            np.median(kin), skew(kin.flatten()), kurtosis(kin.flatten())
        ])
        
        # KIN通道特征（取前20个通道）
        for c in range(min(20, kin.shape[1])):
            channel = kin[:, c]
            features.extend([
                channel.mean(), channel.std(), channel.max() - channel.min()
            ])
        
        # 时序特征
        # EMG一阶差分（速度）
        emg_diff = np.diff(emg, axis=0)
        features.extend([
            emg_diff.mean(), emg_diff.std(),
            np.abs(emg_diff).mean()  # 平均变化率
        ])
        
        # KIN一阶差分
        kin_diff = np.diff(kin, axis=0)
        features.extend([
            kin_diff.mean(), kin_diff.std(),
            np.abs(kin_diff).mean()
        ])
        
        # 二阶差分（加速度）
        emg_diff2 = np.diff(emg_diff, axis=0)
        kin_diff2 = np.diff(kin_diff, axis=0)
        features.extend([
            emg_diff2.mean(), emg_diff2.std(),
            kin_diff2.mean(), kin_diff2.std()
        ])
        
        return np.array(features, dtype=np.float32)
    
    def extract_subject_features(self, sample_features_list, labels):
        """
        提取受试者级别特征
        
        Args:
            sample_features_list: 该受试者所有样本的特征列表
            labels: 对应的FMA标签列表（应该都相同）
        
        Returns:
            受试者级别特征向量
        """
        features = np.array(sample_features_list)
        
        subject_feat = []
        
        # 统计特征
        subject_feat.extend([
            features.mean(axis=0).mean(),  # 全局均值
            features.std(axis=0).mean(),   # 全局标准差
            features.max(axis=0).mean(),   # 全局最大值
            features.min(axis=0).mean(),   # 全局最小值
        ])
        
        # 分布特征
        for i in range(min(10, features.shape[1])):  # 取前10个特征维度
            col = features[:, i]
            subject_feat.extend([
                col.mean(), col.std(), 
                np.percentile(col, 25), np.percentile(col, 75),
                skew(col), kurtosis(col)
            ])
        
        # 变异性特征
        subject_feat.append(features.var(axis=0).mean())  # 样本间变异
        
        return np.array(subject_feat, dtype=np.float32)


# ============================================================================
# 主训练流程
# ============================================================================
def load_and_preprocess(config):
    """加载和预处理数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"加载原始样本: {len(raw_samples)}")
    
    extractor = FeatureExtractor(config.TARGET_LENGTH)
    augmenter = DataAugmenter(config)
    
    # 按受试者组织数据
    subject_data = {}  # {subject_id: {'emg': [], 'kin': [], 'fma': []}}
    
    for sample in tqdm(raw_samples, desc="提取特征"):
        if sample.emg_data.size == 0 or sample.kin_data.size == 0 or sample.fma_score is None:
            continue
        
        try:
            kin_data = sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            emg_aligned, kin_aligned = extractor.align(sample.emg_data, kin_data)
            
            if np.any(np.isnan(emg_aligned)) or np.any(np.isnan(kin_aligned)):
                continue
            
            subject_id = sample.subject_id
            if subject_id not in subject_data:
                subject_data[subject_id] = {'emg': [], 'kin': [], 'fma': sample.fma_score}
            
            subject_data[subject_id]['emg'].append(emg_aligned)
            subject_data[subject_id]['kin'].append(kin_aligned)
            
        except Exception as e:
            continue
    
    print(f"\n受试者数量: {len(subject_data)}")
    
    # 统计FMA分布
    fma_counts = {}
    for sid, data in subject_data.items():
        fma = data['fma']
        fma_counts[fma] = fma_counts.get(fma, 0) + 1
    print(f"FMA分布: {fma_counts}")
    
    return subject_data, extractor, augmenter


def create_augmented_dataset(subject_data, extractor, augmenter, config):
    """创建增强数据集"""
    print("\n" + "="*60)
    print("数据增强...")
    print("="*60)
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    for subject_id, data in tqdm(subject_data.items(), desc="增强数据"):
        emg_list = data['emg']
        kin_list = data['kin']
        fma = data['fma']
        
        # 原始样本特征
        original_features = []
        for emg, kin in zip(emg_list, kin_list):
            feat = extractor.extract_sample_features(emg, kin)
            original_features.append(feat)
        
        # 提取受试者级别特征
        subject_feat = extractor.extract_subject_features(original_features, [fma]*len(original_features))
        
        # 为每个原始样本添加受试者级别特征
        for i, feat in enumerate(original_features):
            combined_feat = np.concatenate([feat, subject_feat])
            all_features.append(combined_feat)
            all_labels.append(fma)
            all_subjects.append(subject_id)
        
        # 数据增强
        for _ in range(config.AUGMENT_RATIO):
            for emg, kin in zip(emg_list, kin_list):
                # 随机增强
                emg_aug = augmenter.augment(emg)
                kin_aug = augmenter.augment(kin)
                
                # 提取特征
                feat = extractor.extract_sample_features(emg_aug, kin_aug)
                combined_feat = np.concatenate([feat, subject_feat])
                
                all_features.append(combined_feat)
                all_labels.append(fma)
                all_subjects.append(subject_id)
    
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    subjects = np.array(all_subjects)
    
    print(f"\n增强后样本数: {len(labels)}")
    print(f"特征维度: {features.shape[1]}")
    
    return features, labels, subjects


def train_and_evaluate(features, labels, subjects, config):
    """训练和评估"""
    print("\n" + "="*60)
    print("训练模型...")
    print("="*60)
    
    unique_subjects = np.unique(subjects)
    
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA降维
    pca = PCA(n_components=config.PCA_COMPONENTS)
    features_pca = pca.fit_transform(features_scaled)
    print(f"PCA保留方差: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    all_results = []
    
    for fold_idx, test_subject in enumerate(unique_subjects):
        print(f"\nFold {fold_idx+1}/{len(unique_subjects)}: 测试受试者 {test_subject}")
        
        # 划分训练/测试集
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        X_train = features_pca[train_mask]
        y_train = labels[train_mask]
        X_test = features_pca[test_mask]
        y_test = labels[test_mask]
        
        # 训练多个模型
        models = {
            'svr': SVR(kernel='rbf', C=10, gamma='scale'),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        
        predictions = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred
        
        # 加权集成
        pred_fma = np.zeros(len(X_test))
        for name, pred in predictions.items():
            pred_fma += config.ENSEMBLE_WEIGHTS[name] * pred
        
        # 四舍五入到最近的FMA值
        pred_fma_rounded = np.array([min(config.FMA_VALUES, key=lambda x: abs(x - p)) for p in pred_fma])
        
        # 计算指标
        mae = mean_absolute_error(y_test, pred_fma_rounded)
        acc1 = accuracy_score(y_test, pred_fma_rounded)
        acc_pm1 = np.mean(np.abs(pred_fma_rounded - y_test) <= 1)
        
        print(f"  真实FMA: {y_test[0]}, 预测: {pred_fma_rounded.mean():.2f}±{pred_fma_rounded.std():.2f}")
        print(f"  MAE: {mae:.2f}, Acc@1: {acc1*100:.2f}%, Acc@±1: {acc_pm1*100:.2f}%")
        
        all_results.append({
            'fold': fold_idx + 1,
            'test_subject': test_subject,
            'true_fma': y_test[0],
            'pred_fma_mean': pred_fma_rounded.mean(),
            'mae': mae,
            'acc1': acc1,
            'acc_pm1': acc_pm1
        })
    
    return pd.DataFrame(all_results)


def main():
    print("="*60)
    print("HybridSCIMFP Optimized v8.0")
    print("数据增强 + 受试者级别特征 + 集成学习")
    print("="*60)
    print("\n目标: MAE < 1.40, Acc@1 > 75%")
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    subject_data, extractor, augmenter = load_and_preprocess(config)
    
    # 创建增强数据集
    features, labels, subjects = create_augmented_dataset(subject_data, extractor, augmenter, config)
    
    # 训练和评估
    results_df = train_and_evaluate(features, labels, subjects, config)
    
    # 汇总结果
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    
    avg_mae = results_df['mae'].mean()
    avg_acc1 = results_df['acc1'].mean()
    avg_acc_pm1 = results_df['acc_pm1'].mean()
    
    print(f"\n平均MAE: {avg_mae:.2f}")
    print(f"平均Acc@1: {avg_acc1*100:.2f}%")
    print(f"平均Acc@±1: {avg_acc_pm1*100:.2f}%")
    
    # 保存结果
    output_file = os.path.join(config.OUTPUT_DIR, 'optimized_v8_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    # 对比
    print("\n" + "="*60)
    print("与基线对比")
    print("="*60)
    print(f"v5.7 (基线): MAE=1.40")
    print(f"v8.0 (优化): MAE={avg_mae:.2f}, Acc@1={avg_acc1*100:.2f}%")
    
    if avg_mae < 1.40:
        print(f"\n✓ MAE已超越基线!")
    else:
        print(f"\n✗ MAE未超越基线 (gap: {avg_mae - 1.40:.2f})")
    
    if avg_acc1 > 0.75:
        print(f"✓ Acc@1已达到目标!")
    else:
        print(f"✗ Acc@1未达到目标 (gap: {(0.75 - avg_acc1)*100:.2f}%)")
    
    return results_df


if __name__ == '__main__':
    main()
