# DeepCS — 解耦力导向图布局

基于解耦力预测 + 迭代 ASGD 求解器的图布局学习系统。支持多种损失项（KL、LNP、AR、EU），包含完整的 HPO 超参搜索与消融评测流水线。

## 部署流程

### 第一步：克隆仓库

```bash
git clone https://github.com/dogcatdogdog/deep_cs.git
cd deep_cs
```

### 第二步：安装环境

```bash
# 方式 A：Conda（推荐，含 CUDA 12.4）
conda env create -f environment.yml
conda activate deepcs

# 方式 B：pip
pip install -r requirements.txt
```

### 第三步：恢复数据文件

训练与评测依赖的外部数据未存储在 Git 仓库中（文件过大），需要从
[GitHub Releases](https://github.com/dogcatdogdog/deep_cs/releases)
下载并放置到对应位置。

#### 3.1 下载文件

| Release 文件 | 大小 | 内容 |
|---|---|---|
| `rome-graphml.tgz` | 4.7 MB | 完整 Rome 图集（11,536 个 `.graphml`） |
| `benchmark_test_0312.zip` | 2.8 MB | 46 个 `.json` 评测用例（默认 benchmark） |
| `rome_1000.zip` | 4.0 MB | 1,000 个 `.graphml` 评测用例（大规模 benchmark） |
| `checkpoints.tar.gz` | ~43 MB | 预训练模型（可选，跳过训练直接评测） |
| `eval_out.tar.gz` | ~10 GB | 预计算评测结果（可选，直接获得论文图表） |

#### 3.2 放置到正确路径

```bash
# --- 训练数据 ---
# 将 rome-graphml.tgz 解压到 data/raw/
tar -xzf rome-graphml.tgz -C data/raw/
# 结果：data/raw/rome/ 内有 11,536 个 .graphml 文件

# --- 评测数据 ---
# 将 benchmark_test_0312 解压到 eval/
unzip benchmark_test_0312.zip -d eval/
# 结果：eval/benchmark_test_0312/ 内有 46 个 .json 文件

# 将 rome_1000 解压到 eval/
unzip rome_1000.zip -d eval/
# 结果：eval/rome_1000/ 内有 1000 个 .graphml 文件

# --- 预训练模型（可选）---
tar -xzf checkpoints.tar.gz -C logs/
# 结果：logs/ablation/ 和 logs/HPO/ 下有各实验的 best_model.pth
```

#### 3.3 目录结构确认

```
deep_cs/
├── data/
│   ├── raw/
│   │   └── rome/            ← rome-graphml.tgz 解压 (11,536 .graphml)
│   └── processed/           (自动生成 .pt 缓存)
├── eval/
│   ├── benchmark_test_0312/ ← benchmark_test_0312.zip 解压 (46 .json)
│   └── rome_1000/           ← rome_1000.zip 解压 (1000 .graphml)
├── logs/                    ← checkpoints.tar.gz 解压 (可选)
├── configs/                 配置文件
├── models/                  模型定义
├── losses/                  损失函数
└── hpo_results/             HPO 结果
```

> **说明：**
> - `dataset_decoupled_v10.py` 从 `data/raw/rome/` 按字母序取前 500 个图用于训练
> - `batch_benchmark_pipeline_v2.py` 默认读 `benchmark_test_0312`，可通过脚本底部 `INPUT_DIRECTORY` 变量切换为 `rome_1000`
> - SuiteSparse 矩阵由 `fetch_suitesparse.py` 首次运行时自动下载到 `data/raw/suitesparse/`，无需手动处理

### 第四步：验证安装

```bash
# 确认训练数据就绪
ls data/raw/rome/*.graphml | wc -l    # 应输出 11536

# 确认评测数据就绪
ls eval/benchmark_test_0312/*.json | wc -l   # 应输出 46
ls eval/rome_1000/*.graphml | wc -l          # 应输出 1000（如已下载）

# 测试数据集加载（会自动生成 processed cache）
python -c "from data.dataset_decoupled_v10 import ProbForceDataset; \
           d = ProbForceDataset(root='./data'); \
           print(f'加载成功: {len(d)} 个图')"
```

### 第五步：运行

```bash
# 训练单个损失项
python train_single_loss.py --config configs/kl_config.yaml

# 依次训练全部损失项
python run_all_losses.py

# HPO 超参搜索
python run_hpo_v10_sinar.py

# 消融实验（6 种 GNN × 4 种 Loss）
python run_ablation_matrix.py

# 导出 HPO 结果
python export_hpo_results.py

# 评测（默认 benchmark_test_0312，46 个图）
cd eval
python batch_benchmark_pipeline_v2.py
```

## 项目结构

```
deep_cs/
├── configs/                  YAML 配置（每种 loss / 消融）
│   ├── default.yaml
│   ├── kl_config.yaml
│   ├── lnp_config.yaml
│   ├── ar_config.yaml
│   ├── eu_config.yaml
│   └── ablation_config.yaml
├── data/                     数据集加载 & SuiteSparse 获取
│   ├── dataset_decoupled_v10.py
│   ├── fetch_suitesparse.py
│   ├── raw/                  ← 解压训练数据至此
│   └── processed/            ← 自动生成缓存
├── eval/                     评测流水线、指标、布局引擎
│   ├── batch_benchmark_pipeline_v2.py
│   ├── eval_utils.py
│   ├── layout.py
│   ├── benchmark_test_0312/  ← 解压评测用例至此
│   └── rome_1000/            ← 解压大规模评测用例至此
├── hpo_results/              HPO 结果数据库 & Pareto 前沿
│   ├── deepcs_v10_sin_ar_hpo.db
│   ├── all_trials_v10_sinar.csv
│   └── pareto_front_v10_sinar.csv
├── logs/                     ← 解压预训练模型至此（可选）
├── losses/                   损失函数实现
│   └── doubled_loss.py
├── models/                   模型定义
│   ├── decoupled_predictor_v2.py
│   └── asgd_force_solver_v2.py
├── utils.py                  工具函数
├── environment.yml           Conda 环境定义
├── requirements.txt          Pip 依赖
└── README.md
```

## 数据说明

| 数据 | 来源 | 放置路径 | 用途 |
|---|---|---|---|
| Rome 图集 | `rome-graphml.tgz` | `data/raw/rome/` | 训练数据 |
| Benchmark 用例 | `benchmark_test_0312.zip` | `eval/benchmark_test_0312/` | 默认评测（46 图） |
| Rome 评测集 | `rome_1000.zip` | `eval/rome_1000/` | 大规模评测（1000 图） |
| SuiteSparse | 自动下载 | `data/raw/suitesparse/` | 训练数据补充 |
| 预训练模型 | `checkpoints.tar.gz` | `logs/` | 跳过训练直接评测 |
| 预计算评测结果 | `eval_out.tar.gz` | `eval_out/` | 直接获得论文图表 |
