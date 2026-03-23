# ESM3 Nanobody CDR3 Optimization

基于 `ESM3_Nanobody_CDR3_Optimization_Project.md` 落地的可运行项目，用于完成纳米抗体 CDR3 重设计、结构质量评估、人源化分析和结果图表生成。

## 项目状态

- [x] CDR3 生成器：支持 ESM3 模型加载
- [x] 结构预测：ESMFold 集成
- [x] 候选评分：多维度综合评分
- [x] CLI 接口：完整命令行工具
- [x] 人源化评估：氨基酸组成分析、免疫原性评分、骆驼特征保留度
- [x] 数据下载：PDB 和 UniProt

## 目录结构

```text
.
├── configs/
│   └── default_config.json       # 默认配置文件
├── data/                         # 数据目录
├── outputs/                      # 输出目录
├── src/
│   └── esm3_nanobody/
│       ├── __init__.py
│       ├── cli.py                # 命令行接口
│       ├── generator.py          # CDR3 生成器
│       ├── scorer.py              # 评分排序
│       ├── structure_predictor.py # 结构预测
│       ├── humanization_analyzer.py # 人源化评估
│       └── data_utils.py         # 数据下载工具
├── tests/                        # 测试目录
├── run_pipeline.sh               # 运行脚本
├── run_pipeline.ps1
├── requirements.txt
└── README.md
```

## 环境配置

### 1. 创建虚拟环境

```bash
# 创建 conda 环境
conda create -n esm3_nanobody python=3.11
conda activate esm3_nanobody

# 或者使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载 ESM3 模型

模型文件应该放在：`E:\teacherstudnet model\esm3_sm_open_v1.pth`

如果路径不同，请在 `configs/default_config.json` 中修改 `model_path`。

## 运行方式

### 1. 查看配置

```bash
cat configs/default_config.json
```

### 2. 运行完整流程

```bash
# Windows PowerShell
.\run_pipeline.ps1

# Linux/Mac
bash run_pipeline.sh
```

或者直接使用 Python：

```powershell
# 激活虚拟环境后
python -m src.esm3_nanobody.cli run --config configs/default_config.json
```

### 3. 下载示例数据

```bash
python -m src.esm3_nanobody.cli download-data --output-dir data/downloaded
```

这会下载：
- PDB: `1MWE`, `2P42`, `5E5E` (VHH 纳米抗体结构)
- UniProt: `P0DTC2` (Spike), `P04626` (HER2)
## 配置说明

### 默认配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_path` | ESM3 模型路径 | `E:/teacherstudnet model/esm3_sm_open_v1.pth` |
| `device` | 运行设备 | `auto` (自动选择 GPU/CPU) |
| `temperature` | 生成温度 | 0.8 |
| `top_k` | Top-K 采样 | 40 |
| `top_p` | Nucleus 采样 | 0.9 |
| `num_candidates` | 生成候选数 | 32 |
| `refinement_rounds` | 优化轮数 | 3 |
| `weights.humanization` | 人源化程度权重 | 0.35 |
| `weights.plddt` | 结构质量权重 | 0.3 |
| `weights.diversity` | 多样性权重 | 0.15 |
| `weights.stability` | 稳定性权重 | 0.15 |

### 框架区序列

可以在配置中指定纳米抗体框架区序列：

```json
{
    "framework_sequence": "QVQLVESGGGLVQAGGSLRLSCAASGRAVYYGATNYWAKGRAPISGRDYWGKQATLVTGASPVHRELYRQNVGGLELNTVGLGLWGGGFVVTGGYNYNYGYTYNYNYGMDV"
}
```

## 输出结果

每次运行会在 `outputs/<run_name>/` 下生成：

| 文件 | 说明 |
|------|------|
| `candidates.json` | 完整候选数据 (JSON) |
| `candidates.csv` | 候选数据 (CSV) |
| `summary.json` | 运行摘要 |
| `top_candidates.txt` | Top 10 候选列表 |
| `figures/analysis.png` | 分析图表 |

## 评分体系

最终得分 = 0.35 × 人源化程度 + 0.35 × 结构质量 + 0.15 × 多样性 + 0.15 × 稳定性

### 各项评分标准

1. **人源化分数**：基于氨基酸组成与人类 CDR3 的相似度
2. **结构质量分数**：基于 pLDDT 预测置信度
3. **多样性分数**：基于序列唯一性和 Levenshtein 距离
4. **稳定性分数**：基于疏水性、胱氨酸、成对脯氨酸等

## 示例框架区序列

常用的纳米抗体框架区序列（来源于不同 PDB）：

- **2P42** (抗 VEGF 纳米抗体)
- **5E5E** (抗 Spike RBD 纳米抗体)
- **1MWE** (基础 VHH 框架)

## 注意事项

1. **ESM3 模型加载**：首次运行时会加载 ~2.8GB 的模型文件，可能需要几分钟
2. **GPU 推荐**：建议使用 GPU 运行以加速结构预测
3. **网络访问**：下载示例数据需要访问 RCSB PDB 和 UniProt

## 故障排除

### 问题：ESM3 模型加载失败

确保：
1. 模型文件存在于指定路径
2. 已正确安装 `esm` 包
3. 有足够的内存（建议 16GB+ RAM）

### 问题：结构预测太慢

可以：
1. 使用 GPU 加速
2. 减少候选数量 (`num_candidates`)
3. 使用 ESMFold 代替 AlphaFold
