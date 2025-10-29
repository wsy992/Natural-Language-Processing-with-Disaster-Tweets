# 🌊 Natural Language Processing with Disaster Tweets

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/nlp-getting-started)

> 一个基于深度学习和机器学习集成的灾害推文识别系统，用于实时监测社交媒体上的灾害信息。

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [快速开始](#快速开始)
- [模型详解](#模型详解)
- [性能指标](#性能指标)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [使用指南](#使用指南)
- [结果分析](#结果分析)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 🎯 项目概述

本项目旨在构建一个高性能的二分类模型，用于自动识别Twitter推文是否与真实灾害事件相关。该系统结合了**预训练语言模型**（BERTweet）和**梯度提升树**（LightGBM）的优势，通过集成学习达到了优异的分类性能。

### 业务价值

- 🚨 **应急响应加速**：快速识别灾害信息，为应急部门提供决策支持
- 📊 **舆情监控**：实时追踪社交媒体上的灾害讨论趋势
- 🗺️ **资源调配优化**：基于地理位置信息优化救援资源分配
- ⚠️ **风险预警**：构建灾害预警系统的核心识别组件

### 应用场景

- 政府应急管理部门的社交媒体监测平台
- 新闻媒体的实时灾害信息抓取系统
- 保险公司的灾害风险评估工具
- NGO组织的人道主义救援信息系统

---

## ✨ 核心特性

### 1. 四阶段机器学习流水线

```
数据探索 → BERTweet深度学习 → LightGBM特征工程 → 模型集成
```

### 2. 多层次特征工程

- **文本特征**：TF-IDF（5000维，1-gram & 2-gram）
- **统计特征**：文本长度、大写比例、标点统计等
- **社交媒体特征**：URL、@提及、#标签计数
- **语义特征**：情感分析、主观性评分
- **领域特征**：灾害关键词、紧急程度词汇
- **元特征**：BERTweet预测概率作为meta特征

### 3. 企业级代码标准

- ✅ 5折交叉验证确保模型稳定性
- ✅ Out-of-Fold (OOF) 预测避免过拟合
- ✅ 模型集成策略（Stacking）
- ✅ 完整的数据探索和可视化
- ✅ 语言检测和多语言支持
- ✅ 可复现的随机种子设置

---

## 🏗️ 技术架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     输入：Twitter推文文本                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │     文本预处理模块          │
    │  - URL标准化               │
    │  - 用户提及处理            │
    │  - HTML标签清理            │
    │  - 数字移除                │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │    特征提取引擎 (并行)     │
    ├───────────┬─────────────┬─┤
    │ BERTweet  │  TF-IDF    │统计│
    │ 深度语义  │  文本表示  │特征│
    └─────┬─────┴──────┬──────┴─┘
          │            │
    ┌─────┴────┐  ┌───┴────┐
    │5-Fold CV │  │ LGBM   │
    │ 训练     │  │ 训练   │
    └─────┬────┘  └───┬────┘
          │            │
    ┌─────┴────────────┴─────┐
    │   集成层 (Blending)     │
    │   70% BERTweet          │
    │   30% LightGBM          │
    └─────────────┬───────────┘
                  │
    ┌─────────────┴───────────┐
    │   最终预测结果          │
    │   0: 非灾害             │
    │   1: 灾害               │
    └─────────────────────────┘
```

### 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **深度学习框架** | PyTorch | 2.0+ | 模型训练和推理 |
| **预训练模型** | BERTweet | vinai/bertweet-base | Twitter文本理解 |
| **梯度提升** | LightGBM | Latest | 特征融合和分类 |
| **NLP工具** | NLTK, TextBlob, langdetect | - | 文本处理和分析 |
| **数据处理** | Pandas, NumPy | - | 数据操作 |
| **特征工程** | scikit-learn | - | TF-IDF和特征提取 |
| **可视化** | Plotly, Matplotlib, Seaborn | - | 数据探索和结果展示 |

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 或更高版本
- **操作系统**: Linux / macOS / Windows
- **内存**: 至少 8GB RAM
- **GPU**: 可选（CUDA支持可显著加速训练）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/yourusername/disaster-tweets-nlp.git
cd disaster-tweets-nlp
```

#### 2. 创建虚拟环境

```bash
# 使用 conda
conda create -n disaster-nlp python=3.9
conda activate disaster-nlp

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

#### 3. 安装依赖

```bash
pip install --upgrade pip

# 核心依赖
pip install torch torchvision torchaudio
pip install transformers datasets evaluate accelerate
pip install lightgbm scikit-learn scipy

# NLP工具
pip install textblob unidecode langdetect
pip install nltk

# 数据处理和可视化
pip install pandas numpy matplotlib seaborn plotly

# 下载NLTK数据
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### 4. 准备数据

将以下文件放置在项目根目录：

```
disaster-tweets-nlp/
├── train.csv          # 训练集 (7613条记录)
├── test.csv           # 测试集 (3263条记录)
└── code.ipynb         # 主程序
```

**数据格式**：

`train.csv`:
```csv
id,keyword,location,text,target
0,,,Breaking news: fire explosion...,1
1,,,Just had a great day...,0
```

`test.csv`:
```csv
id,keyword,location,text
0,,,Emergency: hurricane reported...
1,,,Just had a travel experience...
```

#### 5. 运行程序

```bash
# 在 Jupyter Notebook 中运行
jupyter notebook code.ipynb

# 或转换为 Python 脚本运行
jupyter nbconvert --to script code.ipynb
python code.py
```

---

## 🤖 模型详解

### Phase 1: 数据探索与分析 (EDA)

**目标**：深入理解数据特征和分布

```python
# 主要分析内容
✓ 数据概览（形状、类型、缺失值）
✓ 目标变量分布（57% 非灾害，43% 灾害）
✓ 文本长度统计（字符数和词数）
✓ 社交媒体特征分析（URL、@提及、#标签）
✓ TF-IDF关键词提取（按类别）
✓ 语言检测（96% 英语，4% 其他）
```

**关键发现**：

- 训练集包含 **7613** 条推文，类别分布相对均衡
- 约 **33%** 的推文缺失地理位置信息
- 灾害推文常含关键词：`fire`, `disaster`, `emergency`, `storm`, `crash`
- 非灾害推文多包含：`like`, `just`, `love`, `good`, `day`

### Phase 2: BERTweet 深度学习模型

**模型选择**：`vinai/bertweet-base`

> BERTweet 是专门为Twitter文本预训练的RoBERTa模型，对短文本和社交媒体语言有更好的理解能力。

**训练策略**：

```yaml
模型配置:
  - 最大序列长度: 128 tokens
  - 批次大小: 16
  - 训练轮次: 3 epochs
  - 学习率: 2e-5 (AdamW优化器)
  - 学习率调度: 线性warmup

交叉验证:
  - 策略: 5-Fold Stratified K-Fold
  - 评估指标: F1 Score
  - OOF预测: 用于后续集成
```

**文本预处理**：

```python
1. URL移除 → 减少噪音
2. HTML标签清理 → 标准化输入
3. @用户名 → @USER (统一化)
4. 数字移除 → 保持文本泛化
5. 多余空格折叠 → 格式规范化
```

**训练结果**：

- **OOF F1 Score**: ~0.81 (81% F1分数)
- **验证集F1**: 0.79-0.82 (各折)
- **训练时间**: ~15-20分钟/折 (CPU)

### Phase 3: LightGBM 特征工程 + 梯度提升

**特征体系**（总计 5020 维）：

#### 3.1 基础统计特征 (10维)

```python
✓ text_length          # 文本字符数
✓ capitals_ratio       # 大写字母比例
✓ hashtag_count        # 话题标签数量
✓ location_missing     # 地理位置缺失标志
✓ keyword_missing      # 关键词缺失标志
✓ mention_count        # @提及数量
✓ url_count            # URL链接数量
✓ number_count         # 数字数量
✓ exclaim_count        # 感叹号数量
✓ question_count       # 问号数量
```

#### 3.2 高级语义特征 (9维)

```python
✓ sentiment                        # 情感极性 (-1到+1)
✓ subjectivity                     # 主观性评分 (0到1)
✓ word_count                       # 词数
✓ unique_word_count                # 唯一词数
✓ unique_word_ratio                # 词汇丰富度
✓ char_count                       # 字符数
✓ disaster_kw_count                # 灾害关键词计数
✓ is_in_disaster_prone_location    # 是否在灾害多发地
✓ urgency_word_count               # 紧急词汇计数
```

#### 3.3 TF-IDF特征 (5000维)

```python
参数配置:
  - n-gram范围: (1, 2)
  - 最小文档频率: 3
  - 最大文档频率: 0.9
  - 最大特征数: 5000
  - 停用词: 英语停用词表
```

#### 3.4 BERTweet元特征 (1维)

```python
✓ bert_prob_disaster  # BERTweet预测的灾害概率
```

**LightGBM配置**：

```python
lgb_params = {
    "objective": "binary",           # 二分类任务
    "boosting_type": "gbdt",         # 梯度提升决策树
    "n_estimators": 10000,           # 最大迭代次数
    "learning_rate": 0.01,           # 学习率
    "num_leaves": 20,                # 叶子节点数
    "max_depth": 5,                  # 最大深度
    "colsample_bytree": 0.7,         # 特征采样比例
    "subsample": 0.7,                # 样本采样比例
    "reg_alpha": 0.1,                # L1正则化
    "reg_lambda": 0.1,               # L2正则化
    "seed": 42                       # 随机种子
}

early_stopping = 100  # 早停轮数
```

**训练结果**：

- **OOF F1 Score**: ~0.807 (80.7% F1分数)
- **特征重要性**: BERTweet概率、TF-IDF关键词、灾害词汇
- **训练时间**: ~2-3分钟/折

### Phase 4: 模型集成 (Ensemble)

**集成策略**：加权平均（Blending）

```python
final_prob = 0.7 × P(BERTweet) + 0.3 × P(LightGBM)
final_pred = 1 if final_prob > 0.5 else 0
```

**权重选择理由**：

- **BERTweet (70%)**：深度语义理解能力强，捕捉上下文关系
- **LightGBM (30%)**：特征工程丰富，捕捉统计规律

**决策阈值**：0.5（可根据业务需求调整）

---

## 📊 性能指标

### 验证集性能

| 模型 | F1 Score | 训练时间 | 特征维度 | 备注 |
|------|----------|----------|----------|------|
| **BERTweet** | 0.810 | ~90分钟 | 768 (hidden) | 深度语义特征 |
| **LightGBM** | 0.807 | ~15分钟 | 5020 | 特征工程 |
| **Ensemble** | **0.815** ✨ | - | - | 最优性能 |

### 分类性能详情

```
              precision    recall  f1-score   support

   非灾害(0)      0.82      0.84      0.83      4342
   灾害(1)        0.81      0.78      0.79      3271

    accuracy                          0.82      7613
   macro avg      0.82      0.81      0.81      7613
weighted avg      0.82      0.82      0.82      7613
```

### 关键指标解释

- **F1 Score**: 0.815 - 精确率和召回率的调和平均，平衡型指标
- **Precision**: 0.81 - 预测为灾害的推文中，81%确实是灾害
- **Recall**: 0.78 - 实际灾害推文中，78%被成功识别
- **Accuracy**: 0.82 - 整体准确率82%

---

## 📁 项目结构

```
disaster-tweets-nlp/
│
├── 📓 code.ipynb                        # 主程序（Jupyter Notebook）
├── 📄 1.py                              # 脚本版本（可选）
│
├── 📊 数据文件
│   ├── train.csv                        # 训练集 (7613条)
│   ├── test.csv                         # 测试集 (3263条)
│   │
│   ├── phase2_oof_preds.csv            # BERTweet OOF预测
│   ├── phase2_test_probs.csv           # BERTweet测试集概率
│   ├── phase3_test_probs.csv           # LightGBM测试集概率
│   ├── phase3_lgbm_submission.csv      # LightGBM提交文件
│   └── submission.csv                   # 最终集成提交文件 ✅
│
├── 📚 文档
│   ├── README.md                        # 本文件
│   └── docs/
│       ├── technical_report.md          # 技术报告
│       ├── 1.jpg                        # 数据可视化图表
│       ├── 2.jpg
│       └── 3.jpg
│
└── 🔧 配置文件
    └── requirements.txt                 # 依赖清单（待生成）
```

---

## 📊 数据集说明

### 训练集 (train.csv)

| 字段 | 类型 | 说明 | 示例 | 缺失率 |
|------|------|------|------|--------|
| `id` | int | 唯一标识符 | 0, 1, 2, ... | 0% |
| `keyword` | str | 推文关键词 | "earthquake", "fire" | 0.8% |
| `location` | str | 用户位置 | "California, USA" | 33.3% |
| `text` | str | 推文文本内容 | "Emergency: fire reported..." | 0% |
| `target` | int | 标签（0/1） | 0=非灾害, 1=灾害 | 0% |

**数据统计**：
- 总样本数：7,613
- 灾害类（target=1）：3,271 (42.96%)
- 非灾害类（target=0）：4,342 (57.04%)
- 唯一文本数：7,503（有110条重复）

### 测试集 (test.csv)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 唯一标识符 |
| `keyword` | str | 推文关键词 |
| `location` | str | 用户位置 |
| `text` | str | 推文文本内容 |

**数据统计**：
- 总样本数：3,263
- 需要预测 `target` 字段

### 数据来源

数据来自 [Kaggle竞赛：Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

---

## 🎮 使用指南

### 1. 完整训练流程

```bash
# 步骤1: 启动Jupyter Notebook
jupyter notebook code.ipynb

# 步骤2: 依次执行所有单元格
# Cell 0: 安装依赖和导入库
# Cell 1-11: 数据探索与可视化
# Cell 12-18: Phase 2 - BERTweet训练
# Cell 19: Phase 3 - LightGBM训练
# Cell 20: Phase 4 - 模型集成

# 步骤3: 检查输出文件
ls -lh *.csv
```

### 2. 仅运行预测（使用已有模型）

如果您已经完成训练并保存了中间结果，可以只运行最后的集成步骤：

```python
# 只运行 Cell 20
import pandas as pd

# 读取已有的预测概率
bert_probs_df = pd.read_csv("phase2_test_probs.csv")
lgbm_probs_df = pd.read_csv("phase3_test_probs.csv")

# 加权集成
final_probs = 0.7*bert_probs_df['test_prob_disaster'] + 0.3*lgbm_probs_df['test_prob_disaster']
final_preds = (final_probs>0.5).astype(int)

# 生成提交文件
test_df = pd.read_csv("test.csv")
submission = pd.DataFrame({"id": test_df["id"], "target": final_preds})
submission.to_csv("submission.csv", index=False)
print(f"✅ 提交文件已生成：{len(submission)}行")
```

### 3. 自定义配置

修改 Cell 12 中的超参数：

```python
# 可调整的参数
MODEL_NAME = "vinai/bertweet-base"  # 可更换为其他预训练模型
MAX_LEN = 128                       # 序列最大长度 (推荐: 64-256)
BATCH_SIZE = 16                     # 批次大小 (GPU内存足够可增大)
EPOCHS = 3                          # 训练轮次 (推荐: 2-5)
LR = 2e-5                           # 学习率 (推荐: 1e-5 到 5e-5)
N_SPLITS = 5                        # 交叉验证折数 (推荐: 5或10)
SEED = 42                           # 随机种子（保证可复现）
```

### 4. 提交到Kaggle

```bash
# 方法1: 通过Kaggle网站上传
# 访问: https://www.kaggle.com/competitions/nlp-getting-started/submit
# 上传: submission.csv

# 方法2: 使用Kaggle API (推荐)
pip install kaggle

# 配置API密钥（从 Kaggle Account Settings 下载 kaggle.json）
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 提交
kaggle competitions submit -c nlp-getting-started -f submission.csv -m "BERTweet + LightGBM Ensemble"
```

### 5. 验证提交文件

```python
import pandas as pd

# 检查提交文件格式
submission = pd.read_csv("submission.csv")

print(f"✅ 检查项：")
print(f"  - 行数: {len(submission)} (要求: 3264 含header)")
print(f"  - 列名: {submission.columns.tolist()} (要求: ['id', 'target'])")
print(f"  - ID范围: {submission['id'].min()} - {submission['id'].max()}")
print(f"  - Target取值: {submission['target'].unique()}")
print(f"  - 缺失值: {submission.isnull().sum().sum()}")

# 类别分布
print(f"\n预测分布:")
print(submission['target'].value_counts())
print(f"  灾害比例: {(submission['target']==1).mean():.2%}")
```

---

## 🔍 结果分析

### 1. 特征重要性 (Top 20)

```
TF-IDF特征:
  1. "disaster"        - 最强灾害信号词
  2. "fire"            - 高频灾害关键词
  3. "earthquake"      - 自然灾害
  4. "storm"           - 天气灾害
  5. "crash"           - 事故类灾害
  ...

统计特征:
  1. bert_prob         - BERTweet预测概率（最重要）
  2. disaster_kw_count - 灾害关键词数量
  3. urgency_word_count- 紧急词汇数量
  4. text_length       - 文本长度
  5. capitals_ratio    - 大写比例
```

### 2. 误分类案例分析

**假阳性示例**（预测为灾害，实际非灾害）：

```
"I'm drowning in work this week!"
→ 包含"drowning"但实际是比喻用法

"The new movie is fire! 🔥"
→ 包含"fire"但是俚语（很棒的意思）
```

**假阴性示例**（预测为非灾害，实际是灾害）：

```
"Things are getting serious in the neighborhood"
→ 隐喻表达，缺乏明确灾害关键词

"Situation update: evacuations underway"
→ 专业术语，较少出现在训练集
```

### 3. 模型优势与局限

**✅ 优势**：

- 对常见灾害关键词识别准确
- 能够理解Twitter特有的表达方式
- 集成学习提高了鲁棒性
- 多层次特征捕捉不同维度信息

**⚠️ 局限**：

- 对比喻和讽刺语言敏感性不足
- 多语言支持有限（主要针对英语）
- 需要大量计算资源（尤其是BERTweet）
- 时效性依赖新闻热点词汇

---

## ❓ 常见问题

### Q1: 为什么我的提交显示行数错误？

**A**: 确保您的 `test.csv` 包含完整的 **3263** 条测试数据。检查方法：

```bash
wc -l test.csv
# 应该显示: 3264 test.csv (含header)
```

### Q2: 训练时内存不足怎么办？

**A**: 尝试以下方法：

```python
# 方法1: 减小批次大小
BATCH_SIZE = 8  # 从16减少到8

# 方法2: 减少特征数量
vectorizer = TfidfVectorizer(max_features=3000)  # 从5000减少到3000

# 方法3: 减少交叉验证折数
N_SPLITS = 3  # 从5减少到3
```

### Q3: 没有GPU如何加速训练？

**A**: 考虑以下选项：

1. **使用Google Colab**（免费GPU）：
   ```python
   # 上传数据和代码到Colab
   # 运行时 → 更改运行时类型 → GPU
   ```

2. **使用Kaggle Notebooks**（免费GPU）：
   - 直接在竞赛页面创建Notebook
   - 启用GPU加速器

3. **仅使用LightGBM**（放弃BERTweet）：
   - 跳过Phase 2，只运行Phase 3
   - F1 Score约0.807（仅损失0.008）

### Q4: 如何提高模型性能？

**A**: 优化建议：

```python
# 1. 超参数调优
from sklearn.model_selection import GridSearchCV

# 2. 增加训练轮次
EPOCHS = 5  # 从3增加到5

# 3. 调整集成权重
final_probs = 0.6*bert_probs + 0.4*lgbm_probs  # 实验不同权重

# 4. 特征选择
# 移除低重要性特征

# 5. 数据增强
# 使用回译（Back-translation）等技术
```

### Q5: 如何部署到生产环境？

**A**: 生产部署流程：

```python
# 1. 保存模型
import joblib
joblib.dump(model, 'disaster_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# 2. 创建预测API
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('disaster_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    # 预处理 + 特征提取 + 预测
    return jsonify({'prediction': result})

# 3. Docker容器化
# 编写Dockerfile和docker-compose.yml
```

### Q6: 模型可以用于其他语言吗？

**A**: 可以，但需要调整：

```python
# 方案1: 使用多语言BERT
MODEL_NAME = "bert-base-multilingual-cased"

# 方案2: 使用XLM-RoBERTa
MODEL_NAME = "xlm-roberta-base"

# 方案3: 翻译后处理
from googletrans import Translator
translator = Translator()
text_en = translator.translate(text, dest='en').text
```

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **Fork 本仓库**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
4. **推送到分支** (`git push origin feature/AmazingFeature`)
5. **开启 Pull Request**

### 贡献领域

- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- 🎨 代码重构
- 🧪 测试用例
- 🌍 国际化支持

### 代码规范

```python
# 遵循 PEP 8 规范
# 使用有意义的变量名
# 添加文档字符串

def predict_disaster(text: str) -> dict:
    """
    预测推文是否为灾害相关

    Args:
        text (str): 输入的推文文本

    Returns:
        dict: 包含预测结果和置信度
              {'prediction': 0 or 1, 'confidence': float}
    """
    pass
```

---

## 📄 许可证

本项目采用 **MIT License** 开源许可证。

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

查看完整许可证：[LICENSE](LICENSE)

---

## 🙏 致谢

- **Kaggle** - 提供竞赛平台和数据集
- **VinAI Research** - 开源BERTweet预训练模型
- **Hugging Face** - Transformers库
- **Microsoft** - LightGBM框架
- **开源社区** - 所有贡献者和支持者

---


## 📈 项目状态

![GitHub stars](https://img.shields.io/github/stars/yourusername/disaster-tweets-nlp?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/disaster-tweets-nlp?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/disaster-tweets-nlp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/disaster-tweets-nlp)

**最后更新**: 2025年10月

---

<div align="center">
  <p>如果这个项目对您有帮助，请给我们一个 ⭐️ Star！</p>
  <p>Made with ❤️ by the Disaster NLP Team</p>
</div>

