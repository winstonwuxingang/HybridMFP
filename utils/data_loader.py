"""
数据加载器 - 多模态数据融合平台  # 模块文档字符串，说明模块用途
Data Loader for Multimodal Fusion Platform  # 英文模块名称

支持加载EMG、运动学数据及临床标签  # 模块功能描述
"""

import os  # 导入操作系统模块，用于文件和目录操作
import numpy as np  # 导入NumPy数值计算库，用于数组操作
import pandas as pd  # 导入Pandas数据处理库，用于CSV文件读取
from pathlib import Path  # 从pathlib模块导入Path类，用于路径处理
from typing import Dict, List, Tuple, Optional, Union  # 导入类型提示模块，用于类型注解
from dataclasses import dataclass  # 导入数据类装饰器，用于简化类定义
import re  # 导入正则表达式模块，用于字符串匹配和解析


@dataclass  # 使用dataclass装饰器，自动生成__init__等方法
class SampleData:  # 定义样本数据类，用于存储单个样本的所有信息
    """单个样本数据结构"""  # 类文档字符串

    subject_id: int  # 受试者ID，整型，标识唯一受试者
    task_id: int  # 任务ID，整型，标识不同的运动任务
    repetition: int  # 重复次数，整型，标识同一任务的重复编号
    group: str  # 分组标识，字符串类型，值为healthy/stroke/baseline
    emg_data: np.ndarray  # EMG肌电信号数据，NumPy数组，形状为(time_steps, 12)
    kin_data: np.ndarray  # 运动学数据，NumPy数组，形状为(time_steps, 63)
    fma_score: Optional[float] = None  # Fugl-Meyer评分，可选浮点数，仅中风患者有此评分

    @property  # 定义属性装饰器，将方法转换为只读属性
    def sample_id(self) -> str:  # 定义sample_id属性方法，返回样本唯一标识字符串
        return f"{self.subject_id}_{self.task_id}_{self.repetition}"  # 返回格式化的样本ID字符串


class DataLoader:  # 定义数据加载器类，负责加载多模态数据
    """多模态数据加载器"""  # 类文档字符串

    def __init__(self, data_root: Union[str, Path]):  # 初始化方法，接收数据根目录参数
        """
        初始化数据加载器  # 方法文档字符串开始

        Args:  # 参数说明
            data_root: 数据集根目录  # data_root参数的说明
        """
        self.data_root = Path(data_root)  # 将数据根目录转换为Path对象并存储为实例属性
        self.clinical_labels = self._load_clinical_labels()  # 调用私有方法加载临床标签数据

    def _load_clinical_labels(self) -> Dict[int, int]:  # 定义加载临床标签的私有方法，返回字典类型
        """加载临床标签 (FMA评分)"""  # 方法文档字符串
        labels_path = self.data_root / "Metadata" / "MHH_Clinical_Labels.csv"  # 构建临床标签文件的完整路径
        if not labels_path.exists():  # 检查临床标签文件是否存在
            print(f"警告: 临床标签文件不存在: {labels_path}")  # 文件不存在时打印警告信息
            return {}  # 返回空字典

        df = pd.read_csv(labels_path)  # 使用Pandas读取CSV文件为DataFrame
        labels = {}  # 初始化空字典用于存储标签数据
        for _, row in df.iterrows():  # 遍历DataFrame的每一行
            subject_id = int(row["Subject_ID"].replace("S", ""))  # 从Subject_ID列提取受试者ID，去除"S"前缀并转为整型
            labels[subject_id] = int(row["FMA_UE_Score"])  # 将受试者ID和对应的FMA评分存入字典

        return labels  # 返回标签字典

    def _parse_filename(self, filename: str) -> Tuple[int, int, int]:  # 定义解析文件名的私有方法
        """
        解析文件名  # 方法文档字符串开始

        Args:  # 参数说明
            filename: 文件名 (如 "10_1_1.csv")  # filename参数的说明

        Returns:  # 返回值说明
            (subject_id, task_id, repetition)  # 返回包含三个ID的元组
        """
        match = re.match(r"(\d+)_(\d+)_(\d+)\.csv", filename)  # 使用正则表达式匹配文件名格式
        if match:  # 如果匹配成功
            return int(match.group(1)), int(match.group(2)), int(match.group(3))  # 返回提取的三个整数值
        return None, None, None  # 匹配失败时返回三个None

    def load_emg_data(self, filepath: Path) -> np.ndarray:  # 定义加载EMG数据的方法
        """
        加载EMG数据  # 方法文档字符串开始

        Args:  # 参数说明
            filepath: EMG文件路径  # filepath参数的说明

        Returns:  # 返回值说明
            EMG数据 (time_steps, 12)  # 返回形状为(时间步数, 12通道)的数组
        """
        try:  # 尝试执行以下代码块
            # EMG数据无表头，直接读取数值  # 注释说明EMG数据文件特点
            data = pd.read_csv(filepath, header=None)  # 读取CSV文件，不使用表头
            values = data.values  # 将DataFrame转换为NumPy数组
            # Handle NaN values  # 注释说明接下来处理缺失值
            if np.isnan(values).any():  # 检查数组中是否存在NaN值
                col_means = np.nanmean(values, axis=0)  # 计算每列的均值（忽略NaN）
                nan_indices = np.where(np.isnan(values))  # 获取所有NaN值的索引位置
                values[nan_indices] = np.take(col_means, nan_indices[1])  # 用对应列的均值填充NaN值
                values = np.nan_to_num(values, nan=0.0)  # 将剩余的NaN值替换为0.0
            return values  # 返回处理后的数值数组
        except Exception as e:  # 捕获所有异常
            print(f"加载EMG数据失败: {filepath}, 错误: {e}")  # 打印错误信息
            return None  # 发生异常时返回None

    def load_kin_data(self, filepath: Path) -> np.ndarray:  # 定义加载运动学数据的方法
        """
        加载运动学数据  # 方法文档字符串开始

        Args:  # 参数说明
            filepath: 运动学数据文件路径  # filepath参数的说明

        Returns:  # 返回值说明
            运动学数据 (time_steps, 63)  # 返回形状为(时间步数, 63特征)的数组
        """
        try:  # 尝试执行以下代码块
            # 运动学数据有表头，跳过前两行  # 注释说明运动学数据文件特点
            data = pd.read_csv(filepath, skiprows=2, header=None)  # 读取CSV文件，跳过前两行，不使用表头
            # Replace non-numeric values with NaN  # 注释说明处理非数值数据
            data = data.apply(pd.to_numeric, errors="coerce")  # 将所有数据转换为数值类型，非数值转为NaN
            values = data.values.astype(np.float64)  # 将DataFrame转换为float64类型的NumPy数组
            # Handle NaN values: replace with column mean or 0  # 注释说明NaN处理策略
            if np.isnan(values).any():  # 检查数组中是否存在NaN值
                col_means = np.nanmean(values, axis=0)  # 计算每列的均值（忽略NaN）
                # Replace NaN in col_means with 0  # 注释说明处理均值中的NaN
                col_means = np.nan_to_num(col_means, nan=0.0)  # 将均值中的NaN替换为0.0
                nan_indices = np.where(np.isnan(values))  # 获取所有NaN值的索引位置
                values[nan_indices] = np.take(col_means, nan_indices[1])  # 用对应列的均值填充NaN值
                # If still has NaN, replace with 0  # 注释说明最终NaN处理
                values = np.nan_to_num(values, nan=0.0)  # 将剩余的NaN值替换为0.0
            return values  # 返回处理后的数值数组
        except Exception as e:  # 捕获所有异常
            print(f"加载运动学数据失败: {filepath}, 错误: {e}")  # 打印错误信息
            return None  # 发生异常时返回None

    def load_sample(  # 定义加载单个样本的方法
        self, subject_id: int, task_id: int, repetition: int, group: str  # 方法参数：受试者ID、任务ID、重复次数、分组
    ) -> Optional[SampleData]:  # 返回类型为可选的SampleData对象
        """
        加载单个样本  # 方法文档字符串开始

        Args:  # 参数说明
            subject_id: 受试者ID  # subject_id参数说明
            task_id: 任务ID  # task_id参数说明
            repetition: 重复次数  # repetition参数说明
            group: 分组 (healthy/stroke/baseline)  # group参数说明

        Returns:  # 返回值说明
            SampleData对象  # 返回SampleData实例
        """
        # 确定数据目录  # 注释说明接下来根据分组确定数据路径
        if group == "healthy":  # 如果分组为健康受试者
            emg_dir = self.data_root / "Healthy" / "ALL_EMG"  # 设置健康组的EMG数据目录
            kin_dir = self.data_root / "Healthy" / "ALL_KIN"  # 设置健康组的运动学数据目录
        elif group == "stroke":  # 如果分组为中风患者
            emg_dir = self.data_root / "Stroke" / "All_EMG"  # 设置中风组的EMG数据目录
            kin_dir = self.data_root / "Stroke" / "All_KIN"  # 设置中风组的运动学数据目录
        elif group == "baseline":  # 如果分组为基线数据
            # 基线数据在Baseline目录，文件名格式为 S{subject_id}_static.csv  # 注释说明基线数据特点
            emg_dir = self.data_root / "Baseline"  # 设置基线数据的目录
            kin_dir = None  # 基线数据可能只有静态数据，运动学目录设为None
        else:  # 其他未知分组
            print(f"未知分组: {group}")  # 打印错误信息
            return None  # 返回None

        # 构建文件名  # 注释说明接下来构建数据文件名
        if group == "stroke":  # 如果是中风组数据
            # Stroke数据文件名格式: S{subject_id}_{task_id}_{repetition}.csv  # 注释说明中风组文件命名格式
            filename = f"S{subject_id}_{task_id}_{repetition}.csv"  # 构建中风组文件名，带S前缀
        else:  # 其他分组
            filename = f"{subject_id}_{task_id}_{repetition}.csv"  # 构建标准格式文件名

        # 加载EMG数据  # 注释说明接下来加载EMG数据
        emg_path = emg_dir / filename  # 构建EMG文件的完整路径
        if not emg_path.exists():  # 检查EMG文件是否存在
            # 尝试其他命名格式  # 注释说明尝试备用文件名
            emg_path = emg_dir / f"{subject_id}_{task_id}_{repetition}.csv"  # 使用不带S前缀的文件名作为备选

        emg_data = None  # 初始化EMG数据为None
        if emg_path.exists():  # 如果EMG文件存在
            emg_data = self.load_emg_data(emg_path)  # 调用load_emg_data方法加载EMG数据

        # 加载运动学数据  # 注释说明接下来加载运动学数据
        kin_data = None  # 初始化运动学数据为None
        if kin_dir:  # 如果运动学数据目录存在
            kin_path = kin_dir / filename  # 构建运动学数据文件的完整路径
            if kin_path.exists():  # 如果运动学数据文件存在
                kin_data = self.load_kin_data(kin_path)  # 调用load_kin_data方法加载运动学数据

        if emg_data is None and kin_data is None:  # 如果两种数据都不存在
            return None  # 返回None

        # 获取FMA评分  # 注释说明获取临床评分
        fma_score = self.clinical_labels.get(subject_id, None)  # 从临床标签字典中获取该受试者的FMA评分

        return SampleData(  # 创建并返回SampleData对象
            subject_id=subject_id,  # 设置受试者ID
            task_id=task_id,  # 设置任务ID
            repetition=repetition,  # 设置重复次数
            group=group,  # 设置分组
            emg_data=emg_data if emg_data is not None else np.array([]),  # 设置EMG数据，不存在则使用空数组
            kin_data=kin_data if kin_data is not None else np.array([]),  # 设置运动学数据，不存在则使用空数组
            fma_score=fma_score,  # 设置FMA评分
        )

    def load_all_samples(  # 定义加载所有样本的方法
        self,  # 实例自身参数
        group: str = "healthy",  # 分组参数，默认为健康组
        max_subjects: Optional[int] = None,  # 最大受试者数量参数，用于调试，默认无限制
        max_tasks: Optional[int] = None,  # 最大任务数量参数，默认无限制
        max_repetitions: Optional[int] = None,  # 最大重复次数参数，默认无限制
    ) -> List[SampleData]:  # 返回SampleData对象列表
        """
        加载所有样本  # 方法文档字符串开始

        Args:  # 参数说明
            group: 分组 (healthy/stroke/baseline)  # group参数说明
            max_subjects: 最大受试者数量 (用于调试)  # max_subjects参数说明
            max_tasks: 最大任务数量  # max_tasks参数说明
            max_repetitions: 最大重复次数  # max_repetitions参数说明

        Returns:  # 返回值说明
            样本列表  # 返回样本列表
        """
        samples = []  # 初始化空列表用于存储所有样本

        # 确定受试者ID列表  # 注释说明接下来确定受试者ID范围
        if group == "healthy":  # 如果是健康组
            subject_ids = [10, 11, 13, 14, 15, 16]  # 健康受试者ID列表
        elif group == "stroke":  # 如果是中风组
            subject_ids = list(range(1, 22))  # 中风受试者ID范围1-21
        elif group == "baseline":  # 如果是基线组
            subject_ids = list(range(1, 22))  # 基线受试者ID范围1-21
        else:  # 其他未知分组
            raise ValueError(f"未知分组: {group}")  # 抛出值错误异常

        if max_subjects:  # 如果设置了最大受试者数量限制
            subject_ids = subject_ids[:max_subjects]  # 截取指定数量的受试者ID

        # 默认任务和重复次数  # 注释说明默认的任务和重复次数设置
        num_tasks = max_tasks if max_tasks else 30  # 设置任务数量，默认30个任务
        num_repetitions = max_repetitions if max_repetitions else 3  # 设置重复次数，默认每个任务重复3次

        # 加载样本  # 注释说明开始加载样本
        for subject_id in subject_ids:  # 遍历每个受试者ID
            for task_id in range(1, num_tasks + 1):  # 遍历每个任务ID（1到num_tasks）
                for rep in range(1, num_repetitions + 1):  # 遍历每次重复（1到num_repetitions）
                    sample = self.load_sample(subject_id, task_id, rep, group)  # 加载单个样本
                    if sample and (  # 如果样本存在且满足条件
                        sample.emg_data.size > 0 or sample.kin_data.size > 0  # EMG数据或运动学数据非空
                    ):
                        samples.append(sample)  # 将有效样本添加到列表

        print(f"加载 {group} 组数据: {len(samples)} 个样本")  # 打印加载结果统计
        return samples  # 返回样本列表

    def get_data_statistics(self, samples: List[SampleData]) -> Dict:  # 定义获取数据统计信息的方法
        """
        获取数据统计信息  # 方法文档字符串开始

        Args:  # 参数说明
            samples: 样本列表  # samples参数说明

        Returns:  # 返回值说明
            统计信息字典  # 返回包含统计信息的字典
        """
        stats = {  # 初始化统计信息字典
            "total_samples": len(samples),  # 总样本数量
            "subjects": set(),  # 受试者ID集合（使用集合避免重复）
            "emg_shapes": [],  # EMG数据形状列表
            "kin_shapes": [],  # 运动学数据形状列表
            "fma_scores": [],  # FMA评分列表
        }

        for sample in samples:  # 遍历每个样本
            stats["subjects"].add(sample.subject_id)  # 将受试者ID添加到集合
            if sample.emg_data.size > 0:  # 如果EMG数据非空
                stats["emg_shapes"].append(sample.emg_data.shape)  # 记录EMG数据形状
            if sample.kin_data.size > 0:  # 如果运动学数据非空
                stats["kin_shapes"].append(sample.kin_data.shape)  # 记录运动学数据形状
            if sample.fma_score is not None:  # 如果存在FMA评分
                stats["fma_scores"].append(sample.fma_score)  # 记录FMA评分

        stats["subjects"] = sorted(list(stats["subjects"]))  # 将受试者ID集合转换为排序后的列表
        stats["num_subjects"] = len(stats["subjects"])  # 记录受试者总数

        if stats["emg_shapes"]:  # 如果存在EMG数据形状记录
            stats["emg_min_length"] = min(s[0] for s in stats["emg_shapes"])  # 计算EMG数据最小长度
            stats["emg_max_length"] = max(s[0] for s in stats["emg_shapes"])  # 计算EMG数据最大长度
            stats["emg_avg_length"] = np.mean([s[0] for s in stats["emg_shapes"]])  # 计算EMG数据平均长度

        if stats["kin_shapes"]:  # 如果存在运动学数据形状记录
            stats["kin_min_length"] = min(s[0] for s in stats["kin_shapes"])  # 计算运动学数据最小长度
            stats["kin_max_length"] = max(s[0] for s in stats["kin_shapes"])  # 计算运动学数据最大长度
            stats["kin_avg_length"] = np.mean([s[0] for s in stats["kin_shapes"]])  # 计算运动学数据平均长度

        if stats["fma_scores"]:  # 如果存在FMA评分记录
            stats["fma_mean"] = np.mean(stats["fma_scores"])  # 计算FMA评分均值
            stats["fma_std"] = np.std(stats["fma_scores"])  # 计算FMA评分标准差
            stats["fma_min"] = min(stats["fma_scores"])  # 获取FMA评分最小值
            stats["fma_max"] = max(stats["fma_scores"])  # 获取FMA评分最大值

        return stats  # 返回统计信息字典


class DataPreparer:  # 定义数据准备器类，用于准备模型训练数据
    """数据准备器 - 用于模型训练"""  # 类文档字符串

    def __init__(self, samples: List[SampleData]):  # 初始化方法，接收样本列表
        """
        初始化数据准备器  # 方法文档字符串开始

        Args:  # 参数说明
            samples: 样本列表  # samples参数说明
        """
        self.samples = samples  # 将样本列表存储为实例属性

    def prepare_for_training(  # 定义准备训练数据的方法
        self, target_length: Optional[int] = None, normalize: bool = True  # 参数：目标长度和是否归一化
    ) -> Dict[str, np.ndarray]:  # 返回字典类型，键为字符串，值为NumPy数组
        """
        准备训练数据  # 方法文档字符串开始

        Args:  # 参数说明
            target_length: 目标序列长度 (None则自动确定)  # target_length参数说明
            normalize: 是否归一化  # normalize参数说明

        Returns:  # 返回值说明
            训练数据字典  # 返回包含训练数据的字典
        """
        # 确定目标长度  # 注释说明确定序列目标长度
        if target_length is None:  # 如果未指定目标长度
            # 使用中位数长度  # 注释说明使用中位数作为目标长度
            emg_lengths = [  # 创建EMG数据长度列表
                s.emg_data.shape[0] for s in self.samples if s.emg_data.size > 0  # 提取所有非空EMG数据的时间步数
            ]
            target_length = int(np.median(emg_lengths))  # 计算中位数并转为整型作为目标长度

        # 准备数据数组  # 注释说明准备存储数据的列表
        emg_data_list = []  # 初始化EMG数据列表
        kin_data_list = []  # 初始化运动学数据列表
        labels = []  # 初始化标签列表
        sample_ids = []  # 初始化样本ID列表

        for sample in self.samples:  # 遍历每个样本
            # 处理EMG数据  # 注释说明处理EMG数据
            if sample.emg_data.size > 0:  # 如果EMG数据非空
                emg = self._resize_sequence(sample.emg_data, target_length)  # 调整EMG数据序列长度
                if normalize:  # 如果需要归一化
                    emg = self._normalize(emg)  # 对EMG数据进行归一化处理
                emg_data_list.append(emg)  # 将处理后的EMG数据添加到列表

            # 处理运动学数据  # 注释说明处理运动学数据
            if sample.kin_data.size > 0:  # 如果运动学数据非空
                kin = self._resize_sequence(sample.kin_data, target_length)  # 调整运动学数据序列长度
                if normalize:  # 如果需要归一化
                    kin = self._normalize(kin)  # 对运动学数据进行归一化处理
                kin_data_list.append(kin)  # 将处理后的运动学数据添加到列表

            # 标签  # 注释说明处理标签
            if sample.fma_score is not None:  # 如果存在FMA评分
                labels.append(sample.fma_score)  # 将FMA评分添加到标签列表

            sample_ids.append(sample.sample_id)  # 将样本ID添加到列表

        result = {"sample_ids": sample_ids, "target_length": target_length}  # 创建结果字典，存储样本ID和目标长度

        if emg_data_list:  # 如果EMG数据列表非空
            result["emg_data"] = np.array(emg_data_list)  # 将EMG数据列表转换为NumPy数组并存入结果
            print(f"EMG数据形状: {result['emg_data'].shape}")  # 打印EMG数据的形状信息

        if kin_data_list:  # 如果运动学数据列表非空
            result["kin_data"] = np.array(kin_data_list)  # 将运动学数据列表转换为NumPy数组并存入结果
            print(f"运动学数据形状: {result['kin_data'].shape}")  # 打印运动学数据的形状信息

        if labels:  # 如果标签列表非空
            result["labels"] = np.array(labels)  # 将标签列表转换为NumPy数组并存入结果
            print(f"标签数量: {len(labels)}")  # 打印标签数量

        return result  # 返回结果字典

    def _resize_sequence(self, data: np.ndarray, target_length: int) -> np.ndarray:  # 定义调整序列长度的私有方法
        """
        调整序列长度  # 方法文档字符串开始

        Args:  # 参数说明
            data: 输入序列 (time_steps, features)  # data参数说明
            target_length: 目标长度  # target_length参数说明

        Returns:  # 返回值说明
            调整后的序列  # 返回调整长度后的序列
        """
        current_length = data.shape[0]  # 获取当前序列的时间步数

        if current_length == target_length:  # 如果当前长度等于目标长度
            return data  # 直接返回原数据，无需调整
        elif current_length > target_length:  # 如果当前长度大于目标长度
            # 下采样  # 注释说明执行下采样操作
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)  # 生成均匀分布的采样索引
            return data[indices]  # 根据索引提取数据点，实现下采样
        else:  # 如果当前长度小于目标长度
            # 上采样 (线性插值)  # 注释说明执行上采样操作，使用线性插值
            from scipy.interpolate import interp1d  # 导入SciPy的一维插值函数

            x_old = np.linspace(0, 1, current_length)  # 创建原始序列的归一化坐标
            x_new = np.linspace(0, 1, target_length)  # 创建目标序列的归一化坐标

            result = np.zeros((target_length, data.shape[1]))  # 初始化结果数组，形状为(目标长度, 特征数)
            for i in range(data.shape[1]):  # 遍历每个特征维度
                f = interp1d(x_old, data[:, i], kind="linear")  # 为当前特征创建线性插值函数
                result[:, i] = f(x_new)  # 使用插值函数计算新坐标点的值

            return result  # 返回插值后的结果数组

    def _normalize(self, data: np.ndarray) -> np.ndarray:  # 定义归一化数据的私有方法
        """
        归一化数据 (Z-score)  # 方法文档字符串开始

        Args:  # 参数说明
            data: 输入数据  # data参数说明

        Returns:  # 返回值说明
            归一化后的数据  # 返回Z-score归一化后的数据
        """
        mean = np.mean(data, axis=0, keepdims=True)  # 计算每个特征的均值，保持维度以便广播
        std = np.std(data, axis=0, keepdims=True) + 1e-8  # 计算每个特征的标准差，加小常数避免除零
        return (data - mean) / std  # 执行Z-score归一化：(数据-均值)/标准差

    def split_data(  # 定义划分数据集的方法
        self,  # 实例自身参数
        data: Dict[str, np.ndarray],  # 数据字典参数
        train_ratio: float = 0.7,  # 训练集比例，默认0.7
        val_ratio: float = 0.15,  # 验证集比例，默认0.15
        test_ratio: float = 0.15,  # 测试集比例，默认0.15
        random_seed: int = 42,  # 随机种子，默认42
    ) -> Dict[str, Dict[str, np.ndarray]]:  # 返回嵌套字典类型
        """
        划分训练/验证/测试集  # 方法文档字符串开始

        Args:  # 参数说明
            data: 数据字典  # data参数说明
            train_ratio: 训练集比例  # train_ratio参数说明
            val_ratio: 验证集比例  # val_ratio参数说明
            test_ratio: 测试集比例  # test_ratio参数说明
            random_seed: 随机种子  # random_seed参数说明

        Returns:  # 返回值说明
            划分后的数据字典  # 返回划分后的数据字典
        """
        np.random.seed(random_seed)  # 设置随机种子，确保结果可复现

        n_samples = len(data["sample_ids"])  # 获取样本总数
        indices = np.random.permutation(n_samples)  # 生成随机排列的索引数组

        train_end = int(n_samples * train_ratio)  # 计算训练集结束位置
        val_end = train_end + int(n_samples * val_ratio)  # 计算验证集结束位置

        train_indices = indices[:train_end]  # 提取训练集索引
        val_indices = indices[train_end:val_end]  # 提取验证集索引
        test_indices = indices[val_end:]  # 提取测试集索引

        result = {"train": {}, "val": {}, "test": {}}  # 初始化结果字典，包含三个子字典

        for key in ["emg_data", "kin_data", "labels"]:  # 遍历三种数据类型
            if key in data:  # 如果该数据类型存在于输入数据中
                result["train"][key] = data[key][train_indices]  # 提取训练集数据
                result["val"][key] = data[key][val_indices]  # 提取验证集数据
                result["test"][key] = data[key][test_indices]  # 提取测试集数据

        for split in ["train", "val", "test"]:  # 遍历三个数据集划分
            result[split]["sample_ids"] = [  # 为每个划分设置样本ID列表
                data["sample_ids"][i]  # 根据索引提取样本ID
                for i in (  # 使用条件表达式选择对应的索引数组
                    train_indices  # 如果是训练集
                    if split == "train"  # 判断是否为训练集
                    else val_indices  # 否则使用验证集索引
                    if split == "val"  # 判断是否为验证集
                    else test_indices  # 否则使用测试集索引
                )
            ]

        print(  # 打印数据划分结果
            f"数据划分: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, 测试集 {len(test_indices)}"  # 显示各数据集样本数量
        )

        return result  # 返回划分后的数据字典


if __name__ == "__main__":  # 如果作为主程序运行
    # 测试数据加载  # 注释说明以下为测试代码
    import sys  # 导入系统模块

    sys.path.append(str(Path(__file__).parent.parent))  # 将父目录添加到Python路径
    from config import DATASET_ROOT  # 从config模块导入数据集根目录配置

    print("=" * 60)  # 打印分隔线
    print("多模态数据加载器测试")  # 打印测试标题
    print("=" * 60)  # 打印分隔线

    # 初始化加载器  # 注释说明初始化数据加载器
    loader = DataLoader(DATASET_ROOT)  # 创建DataLoader实例

    # 加载健康受试者数据 (限制数量用于测试)  # 注释说明加载测试数据
    print("\n加载健康受试者数据...")  # 打印提示信息
    healthy_samples = loader.load_all_samples("healthy", max_subjects=2, max_tasks=5)  # 加载健康受试者数据，限制数量

    # 获取统计信息  # 注释说明获取数据统计
    print("\n数据统计信息:")  # 打印提示信息
    stats = loader.get_data_statistics(healthy_samples)  # 获取数据统计信息
    for key, value in stats.items():  # 遍历统计信息字典
        if not isinstance(value, list):  # 如果值不是列表类型
            print(f"  {key}: {value}")  # 打印键值对

    # 准备训练数据  # 注释说明准备训练数据
    print("\n准备训练数据...")  # 打印提示信息
    preparer = DataPreparer(healthy_samples)  # 创建DataPreparer实例
    train_data = preparer.prepare_for_training(target_length=1000)  # 准备训练数据，目标长度1000

    # 划分数据  # 注释说明划分数据集
    print("\n划分数据集...")  # 打印提示信息
    splits = preparer.split_data(train_data)  # 划分训练、验证、测试集

    print("\n数据加载测试完成!")  # 打印测试完成信息
