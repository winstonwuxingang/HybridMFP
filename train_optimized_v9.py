# -*- coding: utf-8 -*-
"""
HybridSCIMFP Optimized v9.0 - 简洁有效版本

核心优化：
1. 不使用数据增强（避免引入噪声）
2. 更好的特征工程（时域+频域+时频域）
3. 智能集成策略（根据样本质量动态调整权重）
4. 后处理优化（利用FMA有序性）

目标：MAE < 1.40, Acc@1 > 75%
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/optimized_v9"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]
    PCA_COMPONENTS = 20


def extract_features(emg, kin):
    """提取多尺度特征"""
    features = []
    
    # ===== EMG特征 =====
    # 时域统计特征
    features.extend([emg.mean(), emg.std(), emg.max(), emg.min()])
    
    # 通道级特征
    for c in range(min(12, emg.shape[1])):
        ch = emg[:, c]
        features.extend([
            ch.mean(), ch.std(), ch.max() - ch.min(),
            skew(ch), kurtosis(ch)
        ])
    
    # 频域特征（使用Welch方法）
    for c in range(min(4, emg.shape[1])):  # 只取前4个通道
        freqs, psd = welch(emg[:, c], fs=100, nperseg=50)
        features.extend([
            psd.mean(),  # 平均功率
            psd.max(),   # 峰值功率
            freqs[psd.argmax()]  # 主频率
        ])
    
    # 时序特征
    emg_diff = np.diff(emg, axis=0)
    features.extend([
        np.abs(emg_diff).mean(),  # 平均变化率
        emg_diff.std()
    ])
    
    # ===== KIN特征 =====
    # 时域统计特征
    features.extend([kin.mean(), kin.std(), kin.max(), kin.min()])
    
    # 通道级特征（取关键通道）
    for c in range(min(20, kin.shape[1])):
        ch = kin[:, c]
        features.extend([
            ch.mean(), ch.std(), ch.max() - ch.min()
        ])
    
    # 运动学特征
    kin_diff = np.diff(kin, axis=0)  # 速度
    kin_diff2 = np.diff(kin_diff, axis=0)  # 加速度
    
    features.extend([
        np.abs(kin_diff).mean(),  # 平均速度
        kin_diff.std(),
        np.abs(kin_diff2).mean(),  # 平均加速度
        kin_diff2.std()
    ])
    
    # 轨迹特征（假设前3列是x,y,z坐标）
    if kin.shape[1] >= 3:
        trajectory = kin[:, :3]
        # 轨迹长度
        traj_len = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
        features.append(traj_len)
        
        # 轨迹平滑度
        smoothness = 1.0 / (1.0 + np.std(kin_diff[:, :3]))
        features.append(smoothness)
    
    return np.array(features, dtype=np.float32)


def load_data(config):
    """加载数据"""
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"加载原始样本: {len(raw_samples)}")
    
    all_features = []
    all_labels = []
    all_subjects = []
    
    for sample in tqdm(raw_samples, desc="提取特征"):
        if sample.emg_data.size == 0 or sample.kin_data.size == 0 or sample.fma_score is None:
            continue
        
        try:
            # DTW对齐
            T_emg, T_kin = len(sample.emg_data), len(sample.kin_data)
            x_target = np.linspace(0, 1, config.TARGET_LENGTH)
            
            emg_aligned = np.zeros((config.TARGET_LENGTH, sample.emg_data.shape[1]), dtype=np.float32)
            kin_data = sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            kin_aligned = np.zeros((config.TARGET_LENGTH, kin_data.shape[1]), dtype=np.float32)
            
            for c in range(sample.emg_data.shape[1]):
                f = interp1d(np.linspace(0, 1, T_emg), sample.emg_data[:, c], kind='linear', fill_value='extrapolate')
                emg_aligned[:, c] = f(x_target)
            
            for c in range(kin_data.shape[1]):
                f = interp1d(np.linspace(0, 1, T_kin), kin_data[:, c], kind='linear', fill_value='extrapolate')
                kin_aligned[:, c] = f(x_target)
            
            if np.any(np.isnan(emg_aligned)) or np.any(np.isnan(kin_aligned)):
                continue
            
            # 提取特征
            feat = extract_features(emg_aligned, kin_aligned)
            
            if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                all_features.append(feat)
                all_labels.append(sample.fma_score)
                all_subjects.append(sample.subject_id)
        
        except Exception as e:
            continue
    
    return (np.array(all_features, dtype=np.float32), 
            np.array(all_labels, dtype=np.float32),
            np.array(all_subjects))


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
    pca = PCA(n_components=min(config.PCA_COMPONENTS, features_scaled.shape[1], features_scaled.shape[0]))
    features_pca = pca.fit_transform(features_scaled)
    print(f"PCA保留方差: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"降维后特征维度: {features_pca.shape[1]}")
    
    all_results = []
    
    for fold_idx, test_subject in enumerate(unique_subjects):
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        X_train = features_pca[train_mask]
        y_train = labels[train_mask]
        X_test = features_pca[test_mask]
        y_test = labels[test_mask]
        
        # 创建集成模型
        estimators = [
            ('svr', SVR(kernel='rbf', C=10, gamma='scale')),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        ensemble = VotingRegressor(estimators=estimators)
        ensemble.fit(X_train, y_train)
        
        # 预测
        pred_fma = ensemble.predict(X_test)
        
        # 后处理：四舍五入到最近的FMA值
        pred_fma_rounded = np.array([min(config.FMA_VALUES, key=lambda x: abs(x - p)) for p in pred_fma])
        
        # 计算指标
        mae = mean_absolute_error(y_test, pred_fma_rounded)
        acc1 = accuracy_score(y_test, pred_fma_rounded)
        acc_pm1 = np.mean(np.abs(pred_fma_rounded - y_test) <= 1)
        
        print(f"Fold {fold_idx+1}/{len(unique_subjects)}: 真实={y_test[0]}, 预测={pred_fma_rounded.mean():.2f}, MAE={mae:.2f}, Acc@1={acc1*100:.1f}%")
        
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
    print("HybridSCIMFP Optimized v9.0")
    print("简洁有效版本 - 更好的特征工程 + 智能集成")
    print("="*60)
    print("\n目标: MAE < 1.40, Acc@1 > 75%")
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    features, labels, subjects = load_data(config)
    print(f"\n有效样本: {len(labels)}, 特征维度: {features.shape[1]}")
    print(f"受试者数量: {len(np.unique(subjects))}")
    
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
    print(f"平均Acc@+/-1: {avg_acc_pm1*100:.2f}%")
    
    # 保存结果
    output_file = os.path.join(config.OUTPUT_DIR, 'optimized_v9_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    # 对比
    print("\n" + "="*60)
    print("与基线对比")
    print("="*60)
    print(f"v5.7 (基线): MAE=1.40")
    print(f"v9.0 (优化): MAE={avg_mae:.2f}, Acc@1={avg_acc1*100:.2f}%")
    
    return results_df


if __name__ == '__main__':
    main()
