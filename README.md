# UCAS_moshishib_homework1

MNIST手写数字分类实验 - 模式识别作业1

## 项目简介

本项目实现了多种机器学习算法在MNIST数据集上的分类对比实验，包括：

### 降维方法
- **PCA (主成分分析)**: 无监督线性降维
- **LDA (线性判别分析)**: 有监督线性降维

### 分类算法
- **QDF (二次判别函数)**:
  - RDA (正则化判别分析)
  - MQDF (改进二次判别函数)
- **KNN (K近邻分类器)**: 使用KD树加速

## 环境配置

### 要求
- Python 3.7+
- numpy
- scikit-learn
- matplotlib
- seaborn
- pandas

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行实验

```bash
python main.py
```

### 自定义参数

```bash
# 指定测试维度
python main.py --dimensions 10 20 30 50 100

# 指定样本数量（使用更多数据）
python main.py --sample-size 20000

# 跳过可视化
python main.py --no-viz
```

### 参数说明
- `--dimensions`: 要测试的降维维度列表（默认: 10 20 30 50 100 200）
- `--sample-size`: 使用的样本数量（默认: 10000，0表示使用全部数据）
- `--no-viz`: 跳过生成可视化图表

## 项目结构

```
.
├── README.md                    # 项目说明
├── REPORT.md                    # 实验报告
├── requirements.txt             # 依赖包列表
├── main.py                      # 主程序入口
├── data_loader.py              # 数据加载模块
├── dimensionality_reduction.py # 降维方法实现
├── classifiers.py              # 分类器实现
├── experiments.py              # 实验框架
└── visualizations.py           # 可视化模块
```

## 输出结果

运行程序后会生成：

1. **控制台输出**:
   - 实验进度信息
   - 准确率对比表
   - 训练时间对比表

2. **可视化图表**:
   - `accuracy_comparison.png`: 准确率对比图
   - `time_comparison.png`: 训练时间对比图
   - `accuracy_heatmap.png`: 准确率热力图

## 实验报告

详细的实验分析和结果讨论请参见 [REPORT.md](REPORT.md)

## 实现细节

### 降维方法

**PCA实现**:
- 数据中心化
- 协方差矩阵特征值分解
- 保留前k个主成分

**LDA实现**:
- 使用scikit-learn的LDA实现
- 最多保留c-1个判别方向（c为类别数）

### 分类器

**QDF实现**:
- RDA: 使用α和β参数正则化协方差矩阵
- MQDF: 只保留协方差矩阵的主要特征值

**KNN实现**:
- 使用scikit-learn的KNN
- KD树加速最近邻搜索

## 数据来源

程序会自动从以下来源下载MNIST数据集：
1. **主要来源**: TensorFlow/Keras数据集 (https://www.tensorflow.org/datasets/catalog/mnist)
   - 训练集: 60,000 样本
   - 测试集: 10,000 样本
2. **备用来源**: OpenML (如果TensorFlow下载失败)
3. **合成数据**: 如果网络不可用，程序会自动生成合成数据用于测试

## 参考文献

- MNIST数据集: http://yann.lecun.com/exdb/mnist/
- TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/mnist
- Friedman, J. H. (1989). Regularized Discriminant Analysis
- Kimura, F., et al. (1987). Modified Quadratic Discriminant Functions

## 作者

LiuZhong4105

## 许可证

MIT License