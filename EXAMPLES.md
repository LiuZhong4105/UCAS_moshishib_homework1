# 运行示例 (Examples)

## 示例1: 快速测试

最快的方式来验证项目是否正常工作：

```bash
python demo.py
```

这将使用合成数据运行完整的实验流程，生成所有可视化图表。

## 示例2: 使用真实MNIST数据的完整实验

如果有网络连接，可以下载真实的MNIST数据：

```bash
# 使用默认参数（10000样本，6个维度）
python main.py

# 使用更多样本以获得更准确的结果
python main.py --sample-size 30000

# 测试更多维度
python main.py --dimensions 5 10 15 20 30 50 75 100 150 200
```

## 示例3: 仅生成结果表格（不生成图表）

```bash
python main.py --no-viz
```

输出示例：
```
========================================================================================================================
EXPERIMENTAL RESULTS - ACCURACY
========================================================================================================================
Dimensions  PCA+QDF(RDA)   PCA+QDF(MQDF)   PCA+KNN     LDA+QDF(RDA)   LDA+QDF(MQDF)   LDA+KNN     
------------------------------------------------------------------------------------------------------------------------
10          0.8234         0.8156          0.8567      0.8234         0.8123          0.8456      
20          0.8756         0.8678          0.8934      0.8234         0.8123          0.8456      
30          0.9012         0.8945          0.9156      0.8234         0.8123          0.8456      
50          0.9234         0.9178          0.9345      0.8234         0.8123          0.8456      
100         0.9456         0.9401          0.9523      0.8234         0.8123          0.8456      
200         0.9534         0.9489          0.9601      0.8234         0.8123          0.8456      
========================================================================================================================
```

## 示例4: 小规模快速实验

在开发或调试时使用：

```bash
python main.py --dimensions 10 30 50 --sample-size 2000 --no-viz
```

这将在几秒钟内完成，适合快速迭代。

## 示例5: 自定义Python脚本

创建自定义实验脚本：

```python
from experiments import run_experiments
from visualizations import create_all_plots

# 运行实验
results = run_experiments(
    dimensions=[10, 20, 30, 50, 100],
    sample_size=15000
)

# 生成可视化
create_all_plots(results)

# 或者手动访问结果
print(f"Best PCA+KNN accuracy: {max(results['pca_knn']):.4f}")
print(f"Best LDA+KNN accuracy: {max(results['lda_knn']):.4f}")
```

## 示例6: 测试单个算法

```python
from data_loader import load_mnist, standardize_data
from dimensionality_reduction import PCA, LDA
from classifiers import QDF, KNN

# 加载数据
X_train, X_test, y_train, y_test = load_mnist(sample_size=5000)
X_train, X_test = standardize_data(X_train, X_test)

# 降维
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 分类
knn = KNN(n_neighbors=5)
knn.fit(X_train_pca, y_train)
accuracy = knn.score(X_test_pca, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## 示例7: 比较不同的正则化参数

```python
from experiments import run_experiments

# 测试不同的RDA参数
alphas = [0.3, 0.5, 0.7]
betas = [0.3, 0.5, 0.7]

for alpha in alphas:
    for beta in betas:
        print(f"\nTesting alpha={alpha}, beta={beta}")
        # 需要修改classifiers.py中的参数
        # 或者创建自定义实验循环
```

## 预期输出

### 控制台输出

实验运行时会显示：
```
================================================================================
Starting MNIST Classification Experiments
================================================================================
Loading MNIST dataset...
Training set: 8000 samples
Test set: 2000 samples
Feature dimension: 784

================================================================================
Testing with 10 dimensions
================================================================================

Applying PCA with 10 components...
Applying LDA with 9 components...

PCA + QDF (RDA)...
  Accuracy: 0.8234, Time: 0.12s
PCA + QDF (MQDF)...
  Accuracy: 0.8156, Time: 0.08s
PCA + KNN...
  Accuracy: 0.8567, Time: 0.34s
...
```

### 生成的文件

1. `accuracy_comparison.png` - 准确率对比图
2. `time_comparison.png` - 训练时间对比图
3. `accuracy_heatmap.png` - 准确率热力图

## 性能基准

在不同配置下的预期运行时间（参考）：

| 样本数 | 维度数 | 运行时间（约） |
|--------|--------|----------------|
| 2,000  | 3      | ~5秒           |
| 5,000  | 6      | ~30秒          |
| 10,000 | 6      | ~60秒          |
| 30,000 | 6      | ~5分钟         |
| 10,000 | 10     | ~10分钟        |

*注：实际时间取决于硬件配置*

## 典型准确率范围（真实MNIST）

基于文献和经验，在真实MNIST数据上的预期准确率：

| 方法 | 维度 | 预期准确率范围 |
|------|------|----------------|
| PCA + KNN | 50 | 0.93-0.95 |
| PCA + KNN | 100 | 0.95-0.97 |
| LDA + KNN | 9 | 0.85-0.88 |
| PCA + QDF | 50 | 0.85-0.90 |
| PCA + QDF | 100 | 0.88-0.92 |

## 故障排除

### 内存错误

```bash
# 减少样本数
python main.py --sample-size 5000

# 减少维度数
python main.py --dimensions 10 20 30
```

### 网络错误（无法下载MNIST）

```bash
# 使用demo脚本和合成数据
python demo.py
```

### 运行时间过长

```bash
# 使用更小的数据集
python main.py --sample-size 3000 --dimensions 10 20 30
```

## 进阶使用

### 批量实验

创建脚本运行多组实验：

```python
# batch_experiments.py
from experiments import run_experiments
import json

sample_sizes = [5000, 10000, 20000]
all_results = {}

for size in sample_sizes:
    print(f"\n=== Testing with {size} samples ===")
    results = run_experiments(
        dimensions=[10, 20, 30, 50],
        sample_size=size
    )
    all_results[size] = results

# 保存结果
with open('batch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
```

### 添加新的分类器

```python
# 在classifiers.py中添加新类
class MyClassifier:
    def fit(self, X, y):
        # 训练逻辑
        pass
    
    def predict(self, X):
        # 预测逻辑
        pass
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
```

然后在实验中使用它。
