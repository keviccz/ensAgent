# 快速部署指南

## 📋 部署前检查清单

### 1. 环境文件准备

**重要**：由于路径编码问题，环境文件需要手动复制到 `envs/` 目录。

请从父目录复制这3个文件：
- `../R_environment.yml` → `envs/R_environment.yml`
- `../PY_environment.yml` → `envs/PY_environment.yml`
- `../PY2_environment.yml` → `envs/PY2_environment.yml`

**Windows PowerShell 操作**：
```powershell
Copy-Item ..\R_environment.yml envs\
Copy-Item ..\PY_environment.yml envs\
Copy-Item ..\PY2_environment.yml envs\
```

**Linux/Mac 操作**：
```bash
cp ../R_environment.yml envs/
cp ../PY_environment.yml envs/
cp ../PY2_environment.yml envs/
```

### 2. 创建 Conda 环境

```bash
# R 环境 (IRIS, BASS, DR-SC, BayesSpace)
conda env create -f envs/R_environment.yml

# PY 环境 (SEDR, GraphST, STAGATE)
conda env create -f envs/PY_environment.yml

# PY2 环境 (stLearn)
conda env create -f envs/PY2_environment.yml
```

### 3. 准备数据

确保你的数据目录结构如下：
```
your_data_directory/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv (可选，包含参考注释)
└── spatial/
    ├── tissue_hires_image.png
    ├── tissue_lowres_image.png
    ├── scalefactors_json.json
    └── tissue_positions_list.csv
```

### 4. 配置运行参数

编辑 `configs/DLPFC_151507.yaml`，更新以下参数：

```yaml
sample_id: "你的样本ID"
data_path: "D:/path/to/your/data"  # 更新为你的数据路径
output_dir: "./output/你的样本ID"
n_clusters: 7  # 预期的聚类数
```

---

## 🚀 运行流程

### 方式1：使用示例脚本（推荐）

**Windows**:
```cmd
cd examples
run_demo.bat
```

**Linux/Mac**:
```bash
cd examples
chmod +x run_demo.sh
./run_demo.sh
```

### 方式2：直接运行 orchestrator

```bash
# 使用配置文件
python orchestrator.py --config configs/DLPFC_151507.yaml

# 或使用命令行参数
python orchestrator.py \
  --data_path D:/path/to/data \
  --sample_id DLPFC_151507 \
  --output_dir ./output/DLPFC_151507 \
  --n_clusters 7
```

---

## 📊 预期输出

成功运行后，你会在 `output/DLPFC_151507/` 看到：

```
output/DLPFC_151507/
├── domains/                         # 8个聚类方法的原始结果
│   ├── IRIS_DLPFC_151507_domain.csv
│   ├── BASS_DLPFC_151507_domain.csv
│   ├── ... (共8个)
│
├── DLPFC_151507_aligned.h5ad        # 对齐后的 AnnData 对象
│
├── spot/                            # Spot级别数据（坐标+domain）
│   ├── IRIS_domain_DLPFC_151507_spot.csv
│   ├── ... (共8个)
│
├── PICTURES/                        # 空间可视化图片
│   ├── IRIS_domain_DLPFC_151507.png
│   ├── ... (共8个)
│
├── DEGs/                            # 差异表达基因
│   ├── IRIS_domain_DLPFC_151507_DEGs.csv
│   ├── ... (共8个)
│
├── PATHWAY/                         # 通路富集分析
│   ├── IRIS_domain_DLPFC_151507_PATHWAY.csv
│   ├── ... (共8个)
│
└── tool_runner_report.json          # 运行总结报告
```

---

## ⚠️ 常见问题

### 问题1：环境创建失败

**症状**：`conda env create` 报错

**解决**：
```bash
# 清除 conda 缓存
conda clean --all

# 重新创建环境
conda env create -f envs/R_environment.yml --force
```

### 问题2：某个方法执行失败

**症状**：Pipeline 显示某方法失败

**解决**：
1. 查看 `output/sample_id/tool_runner_report.json`
2. 确认失败方法需要的环境已正确创建
3. 如果不需要该方法，可在配置中移除
4. 或调低 `min_success` 参数（默认需要5个方法成功）

### 问题3：内存不足

**症状**：程序崩溃或被系统杀死

**解决**：
- 减少同时运行的方法数量
- 增加系统内存
- 在配置中减小 `n_top_genes` 参数

### 问题4：R 脚本找不到

**症状**：Rscript command not found

**解决**：
```bash
# Windows: 确保 Rscript 在 PATH 中
# 激活 R 环境后检查
conda activate R
where Rscript

# Linux/Mac:
conda activate R
which Rscript
```

---

## 🔍 验证安装

运行以下脚本验证所有环境是否正确安装：

```python
# test_environments.py
import subprocess
import sys

envs = {
    'R': 'Rscript --version',
    'PY': 'python --version',
    'PY2': 'python --version'
}

for env_name, cmd in envs.items():
    print(f"\n测试 {env_name} 环境...")
    result = subprocess.run(
        f"conda activate {env_name} && {cmd}",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✓ {env_name} 环境正常")
    else:
        print(f"✗ {env_name} 环境异常")
        print(result.stderr)
```

---

## 📞 获取帮助

如果遇到问题：

1. 检查 `tool_runner_report.json` 获取详细错误信息
2. 查看 [README.md](README.md) 的 Troubleshooting 章节
3. 提交 GitHub Issue（包含错误日志）

---

**祝部署顺利！🎉**

