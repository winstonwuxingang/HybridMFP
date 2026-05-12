# -*- coding: utf-8 -*-
"""
HybridSCIMFP - 策略1：时频特征提取（小波变换）

目标：通过小波变换提取EMG信号的时频特征，捕捉频率随时间的变化

新增特征：
1. 小波分解各层能量
2. 小波分解各层熵
3. 小波分解各层统计特征
4. 与现有特征融合
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
import pywt
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import DataLoader as SCIDataLoader


class Config:
    DATA_DIR = r"C:\Users\Administrator\Desktop\openclaw_cloudcode\sci\dataset\Rocky_Data"
    OUTPUT_DIR = "./outputs/wavelet_v1.0"
    TARGET_LENGTH = 100
    FMA_VALUES = [14, 16, 17, 18, 19, 20]


class DTWAligner:
    def __init__(self, target_length=100):
        self.target_length = target_length
    
    def align(self, emg, kin):
        T_emg = len(emg)
        T_kin = len(kin)
        x_target = np.linspace(0, 1, self.target_length)
        
        x_emg = np.linspace(0, 1, T_emg)
        emg_aligned = np.zeros((self.target_length, emg.shape[1]))
        for i in range(emg.shape[1]):
            f = interp1d(x_emg, emg[:, i], kind='linear', fill_value='extrapolate')
            emg_aligned[:, i] = f(x_target)
        
        x_kin = np.linspace(0, 1, T_kin)
        kin_aligned = np.zeros((self.target_length, kin.shape[1]))
        for i in range(kin.shape[1]):
            f = interp1d(x_kin, kin[:, i], kind='linear', fill_value='extrapolate')
            kin_aligned[:, i] = f(x_target)
        
        return emg_aligned, kin_aligned


class WaveletFeatureExtractor:
    """小波变换特征提取器"""
    
    def __init__(self, wavelet='db4', level=5):
        self.wavelet = wavelet
        self.level = level
    
    def extract_wavelet_features(self, signal):
        """提取单通道信号的小波特征"""
        features = []
        
        try:
            # 小波分解
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            
            for i, coeff in enumerate(coeffs):
                if len(coeff) == 0:
                    features.extend([0, 0, 0, 0, 0])
                    continue
                
                # 能量
                energy = np.sum(coeff ** 2)
                features.append(energy)
                
                # 熵
                prob = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
                entropy_val = -np.sum(prob * np.log2(prob + 1e-10))
                features.append(entropy_val)
                
                # 统计特征
                features.append(np.mean(coeff))
                features.append(np.std(coeff))
                features.append(np.max(np.abs(coeff)))
            
            # 跨层特征
            energies = [np.sum(c**2) for c in coeffs if len(c) > 0]
            if len(energies) > 1:
                total_energy = sum(energies)
                # 能量分布
                energy_ratio = [e / (total_energy + 1e-10) for e in energies]
                features.extend(energy_ratio)
                
                # 能量熵
                energy_prob = np.array(energies) / (total_energy + 1e-10)
                energy_entropy = -np.sum(energy_prob * np.log2(energy_prob + 1e-10))
                features.append(energy_entropy)
            else:
                features.extend([0] * (len(coeffs) + 1))
            
        except Exception as e:
            features = [0] * (6 * 5 + 7)  # 6层 * 5特征 + 跨层特征
        
        return np.array(features, dtype=np.float32)
    
    def extract_emg_wavelet_features(self, emg):
        """提取多通道EMG的小波特征"""
        all_features = []
        
        for ch in range(emg.shape[1]):
            feat = self.extract_wavelet_features(emg[:, ch])
            all_features.append(feat)
        
        # 汇总统计
        all_features = np.array(all_features)
        
        # 均值和标准差
        mean_feat = np.mean(all_features, axis=0)
        std_feat = np.std(all_features, axis=0)
        
        return np.concatenate([mean_feat, std_feat])


class EnhancedFeatureExtractor:
    """增强版特征提取器（继承v5.8.2）"""
    
    def __init__(self, fs_emg=1000, fs_kin=100):
        self.fs_emg = fs_emg
        self.fs_kin = fs_kin
        self.wavelet_extractor = WaveletFeatureExtractor(wavelet='db4', level=5)
    
    # ==================== EMG特征 ====================
    
    def extract_emg_time_features(self, emg):
        """EMG时域特征"""
        features = []
        
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            
            features.extend([
                np.mean(np.abs(x)),
                np.sqrt(np.mean(x**2)),
                np.var(x),
                np.std(x),
                np.max(x),
                np.min(x),
                np.max(x) - np.min(x),
            ])
            
            diff = np.diff(x)
            features.extend([
                np.sum(np.abs(diff)) / len(diff),
                np.mean(np.abs(diff)),
                np.std(diff),
            ])
            
            zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2
            features.append(zero_crossings / len(x))
            
            slope_sign_changes = np.sum(np.abs(np.diff(np.sign(diff)))) / 2
            features.append(slope_sign_changes / len(diff) if len(diff) > 0 else 0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_emg_freq_features(self, emg):
        """EMG频域特征"""
        features = []
        
        freq_bands = {
            'low': (0, 50),
            'mid_low': (50, 100),
            'mid': (100, 200),
            'mid_high': (200, 350),
            'high': (350, 500)
        }
        
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            freqs, psd = welch(x, fs=self.fs_emg, nperseg=min(256, len(x)))
            
            if len(psd) > 0 and np.sum(psd) > 0:
                total_power = np.sum(psd)
                
                for band_name, (low, high) in freq_bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    band_power = np.sum(psd[band_mask]) if band_mask.sum() > 0 else 0
                    features.append(band_power / (total_power + 1e-10))
                
                cumsum = np.cumsum(psd)
                median_freq = freqs[np.searchsorted(cumsum, cumsum[-1] / 2)]
                features.append(median_freq)
                
                mean_freq = np.sum(freqs * psd) / (total_power + 1e-10)
                features.append(mean_freq)
                
                peak_freq = freqs[np.argmax(psd)]
                features.append(peak_freq)
                
                psd_norm = psd / (total_power + 1e-10)
                spec_entropy = entropy(psd_norm + 1e-10)
                features.append(spec_entropy)
            else:
                features.extend([0] * 9)
        
        return np.array(features, dtype=np.float32)
    
    def extract_emg_covariance_features(self, emg):
        """EMG协方差矩阵特征"""
        features = []
        
        cov_matrix = np.cov(emg.T)
        diag = np.diag(cov_matrix)
        features.extend([
            np.mean(diag),
            np.std(diag),
            np.max(diag),
            np.min(diag),
        ])
        
        n_channels = emg.shape[1]
        if n_channels > 1:
            off_diag = cov_matrix[np.triu_indices(n_channels, k=1)]
            features.extend([
                np.mean(np.abs(off_diag)),
                np.std(off_diag),
                np.max(off_diag),
                np.min(off_diag),
            ])
            
            corr_matrix = np.corrcoef(emg.T)
            if not np.isnan(corr_matrix).any():
                off_diag_corr = corr_matrix[np.triu_indices(n_channels, k=1)]
                features.extend([
                    np.mean(off_diag_corr),
                    np.std(off_diag_corr),
                    np.max(off_diag_corr),
                    np.min(off_diag_corr),
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_emg_nonlinear_features(self, emg):
        """EMG非线性特征"""
        features = []
        
        for ch in range(emg.shape[1]):
            x = emg[:, ch]
            features.append(kurtosis(x))
            features.append(skew(x))
            
            if len(x) > 10:
                diff = np.diff(x)
                features.append(np.std(diff) / (np.std(x) + 1e-10))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_emg_features(self, emg):
        """提取所有EMG特征（包含小波特征）"""
        time_feat = self.extract_emg_time_features(emg)
        freq_feat = self.extract_emg_freq_features(emg)
        cov_feat = self.extract_emg_covariance_features(emg)
        nonlinear_feat = self.extract_emg_nonlinear_features(emg)
        
        # 新增：小波特征
        wavelet_feat = self.wavelet_extractor.extract_emg_wavelet_features(emg)
        
        return np.concatenate([time_feat, freq_feat, cov_feat, nonlinear_feat, wavelet_feat])
    
    # ==================== 运动学特征 ====================
    
    def extract_kin_basic_features(self, kin):
        """运动学基础特征"""
        features = []
        features.extend([
            np.mean(kin),
            np.std(kin),
            np.max(kin),
            np.min(kin),
            np.max(kin) - np.min(kin),
        ])
        return np.array(features, dtype=np.float32)
    
    def extract_kin_velocity_features(self, kin):
        """速度特征"""
        features = []
        
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        
        if len(vel_mag) > 0:
            features.extend([
                np.mean(vel_mag),
                np.std(vel_mag),
                np.max(vel_mag),
                np.min(vel_mag),
                np.median(vel_mag),
            ])
            
            vel_sorted = np.sort(vel_mag)
            features.extend([
                vel_sorted[int(len(vel_sorted) * 0.25)],
                vel_sorted[int(len(vel_sorted) * 0.75)],
            ])
            
            vel_diff = np.diff(vel_mag)
            if len(vel_diff) > 0:
                features.extend([
                    np.mean(np.abs(vel_diff)),
                    np.std(vel_diff),
                ])
            else:
                features.extend([0, 0])
        else:
            features.extend([0] * 9)
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_acceleration_features(self, kin):
        """加速度特征"""
        features = []
        
        vel = np.diff(kin, axis=0)
        if len(vel) > 1:
            acc = np.diff(vel, axis=0)
            acc_mag = np.sqrt(np.sum(acc**2, axis=1))
            
            if len(acc_mag) > 0:
                features.extend([
                    np.mean(acc_mag),
                    np.std(acc_mag),
                    np.max(acc_mag),
                    np.min(acc_mag),
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_smoothness_features(self, kin):
        """运动平滑度特征"""
        features = []
        
        if len(kin) > 3:
            vel = np.gradient(kin, axis=0)
            acc = np.gradient(vel, axis=0)
            jerk = np.gradient(acc, axis=0)
            jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
            
            if len(jerk_mag) > 0:
                jerk_cost = np.sum(jerk_mag**2) * 0.01
                smoothness = 1 / (1 + jerk_cost)
                features.append(smoothness)
                features.append(np.mean(jerk_mag))
                features.append(np.std(jerk_mag))
            else:
                features.extend([0, 0, 0])
            
            vel_mag = np.sqrt(np.sum(vel**2, axis=1))
            if len(vel_mag) > 2:
                peaks = 0
                for i in range(1, len(vel_mag) - 1):
                    if vel_mag[i] > vel_mag[i-1] and vel_mag[i] > vel_mag[i+1]:
                        peaks += 1
                features.append(peaks / len(vel_mag))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_trajectory_features(self, kin):
        """轨迹特征"""
        features = []
        
        diff = np.diff(kin, axis=0)
        path_length = np.sum(np.sqrt(np.sum(diff**2, axis=1)))
        features.append(path_length)
        
        if len(kin) > 1:
            direct_dist = np.sqrt(np.sum((kin[-1] - kin[0])**2))
            features.append(direct_dist)
            efficiency = direct_dist / (path_length + 1e-10)
            features.append(efficiency)
        else:
            features.extend([0, 0])
        
        if len(kin) > 2:
            vel = np.diff(kin, axis=0)
            if len(vel) > 1:
                vel_norm = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-10)
                direction_changes = np.diff(vel_norm, axis=0)
                curvature = np.sqrt(np.sum(direction_changes**2, axis=1))
                features.extend([
                    np.mean(curvature),
                    np.std(curvature),
                    np.max(curvature),
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_temporal_features(self, kin):
        """时序动态特征"""
        features = []
        
        T = len(kin)
        key_points = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for p in key_points:
            idx = int(p * T)
            if idx < T:
                features.append(np.mean(kin[idx]))
            else:
                features.append(0)
        
        vel = np.diff(kin, axis=0)
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        if len(vel_mag) > 0:
            threshold = np.mean(vel_mag) * 0.1
            active_ratio = np.sum(vel_mag > threshold) / len(vel_mag)
            features.append(active_ratio)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_kin_features(self, kin):
        """提取所有运动学特征"""
        basic_feat = self.extract_kin_basic_features(kin)
        vel_feat = self.extract_kin_velocity_features(kin)
        acc_feat = self.extract_kin_acceleration_features(kin)
        smooth_feat = self.extract_kin_smoothness_features(kin)
        traj_feat = self.extract_kin_trajectory_features(kin)
        temporal_feat = self.extract_kin_temporal_features(kin)
        
        return np.concatenate([
            basic_feat, vel_feat, acc_feat, 
            smooth_feat, traj_feat, temporal_feat
        ])


def extract_subject_features(sample_features):
    """从样本级别特征提取受试者级别特征"""
    subject_feat = []
    
    for i in range(sample_features.shape[1]):
        feat = sample_features[:, i]
        subject_feat.extend([
            np.mean(feat),
            np.std(feat),
            np.min(feat),
            np.max(feat),
            np.median(feat),
            np.percentile(feat, 25),
            np.percentile(feat, 75),
            kurtosis(feat) if len(feat) > 3 else 0,
            skew(feat) if len(feat) > 3 else 0,
        ])
    
    return np.array(subject_feat, dtype=np.float32)


def load_and_extract_features():
    """加载数据并提取特征"""
    print("="*60)
    print("加载数据并提取特征（含小波变换）...")
    print("="*60)
    
    loader = SCIDataLoader(data_root=Config.DATA_DIR)
    raw_samples = loader.load_all_samples(group="stroke")
    print(f"原始样本: {len(raw_samples)}")
    
    dtw_aligner = DTWAligner(target_length=Config.TARGET_LENGTH)
    feature_extractor = EnhancedFeatureExtractor(fs_emg=1000, fs_kin=100)
    
    subject_samples = defaultdict(list)
    subject_fma = {}
    
    print("\n特征提取中...")
    for sample in raw_samples:
        if sample.emg_data.size == 0 or sample.kin_data.size == 0:
            continue
        if sample.fma_score is None:
            continue
        
        try:
            emg_aligned, kin_aligned = dtw_aligner.align(
                sample.emg_data, 
                sample.kin_data[:, :63] if sample.kin_data.shape[1] >= 63 else sample.kin_data
            )
            
            emg_feat = feature_extractor.extract_emg_features(emg_aligned)
            kin_feat = feature_extractor.extract_kin_features(kin_aligned)
            feat = np.concatenate([emg_feat, kin_feat])
            
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                continue
            
            sid = f"S{sample.subject_id}"
            subject_samples[sid].append(feat)
            subject_fma[sid] = sample.fma_score
            
        except Exception as e:
            continue
    
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
    print(f"样本级特征维度: {feats_array.shape[1] if len(subject_samples) > 0 else 0}")
    print(f"受试者级特征维度: {X.shape[1]}")
    print(f"FMA分布: {dict(pd.Series(y).value_counts().sort_index())}")
    
    return X, y, sids


def train_wavelet_v1():
    print("="*60)
    print("HybridSCIMFP - 策略1：时频特征提取（小波变换）")
    print("="*60)
    print()
    print("新增特征:")
    print("  1. 小波分解各层能量")
    print("  2. 小波分解各层熵")
    print("  3. 小波分解各层统计特征")
    print("  4. 能量分布和能量熵")
    print()
    print("目标: MAE<1.40, Acc@1>50%")
    print("="*60)
    
    # 加载数据
    X, y, sids = load_and_extract_features()
    
    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 特征选择
    print("\n" + "="*60)
    print("特征选择")
    print("="*60)
    
    n_features = X_scaled.shape[1]
    n_select = min(50, n_features - 1)
    
    selector = SelectKBest(f_regression, k=n_select)
    X_selected = selector.fit_transform(X_scaled, y)
    
    print(f"原始特征: {n_features}")
    print(f"选择后特征: {X_selected.shape[1]}")
    
    # PCA降维
    print("\n" + "="*60)
    print("PCA降维")
    print("="*60)
    
    pca = PCA(n_components=min(15, X_selected.shape[1] - 1))
    X_pca = pca.fit_transform(X_selected)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA维度: {X_pca.shape[1]}, 解释方差: {explained_var:.2%}")
    
    # LOSOCV
    print("\n" + "="*60)
    print("LOSOCV训练")
    print("="*60)
    
    loo = LeaveOneOut()
    
    all_preds = []
    all_labels = []
    
    for train_idx, test_idx in loo.split(X_pca):
        X_train, X_test_fold = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Top-3模型
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        knn = KNeighborsRegressor(n_neighbors=3)
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42)
        
        svr.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        
        # 加权集成
        w_svr, w_knn, w_gb = 0.24, 0.22, 0.54
        pred = w_svr * svr.predict(X_test_fold)[0] + w_knn * knn.predict(X_test_fold)[0] + w_gb * gb.predict(X_test_fold)[0]
        
        all_preds.append(pred)
        all_labels.append(y_test[0])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    mae = np.mean(np.abs(all_preds - all_labels))
    
    # 四舍五入到FMA值
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
    
    # 与v5.8.2对比
    print("\n" + "="*60)
    print("与v5.8.2对比")
    print("="*60)
    
    print(f"\n{'版本':<30} {'MAE':<10} {'Acc@1':<10} {'Acc@±1':<10}")
    print("-" * 60)
    print(f"{'v5.8.2 (特征工程)':<30} {'0.96':<10} {'35%':<10} {'75%':<10}")
    print(f"{'策略1 (小波变换)':<30} {mae:<10.2f} {acc:.0%}       {acc_pm1:.0%}")
    
    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame({
        'subject_id': sids,
        'true_fma': all_labels,
        'pred_reg': all_preds,
        'pred_round': [Config.FMA_VALUES[np.argmin(np.abs(np.array(Config.FMA_VALUES) - p))] for p in all_preds],
        'error': all_preds - all_labels,
    })
    df.to_csv(f'{Config.OUTPUT_DIR}/wavelet_v1.0_predictions.csv', index=False)
    
    summary = pd.DataFrame([{
        'version': 'wavelet_v1.0',
        'strategy': '小波变换特征',
        'mae': mae,
        'acc_1': acc,
        'acc_pm1': acc_pm1,
    }])
    summary.to_csv(f'{Config.OUTPUT_DIR}/wavelet_v1.0_summary.csv', index=False)
    
    print(f"\n结果已保存: {Config.OUTPUT_DIR}/")
    
    return {
        'mae': mae,
        'acc': acc,
        'acc_pm1': acc_pm1,
        'preds': all_preds,
        'labels': all_labels,
    }


if __name__ == '__main__':
    result = train_wavelet_v1()
