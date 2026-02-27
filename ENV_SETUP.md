# Conda 环境安装（Tool-runner 用）

Tool-runner 需要 3 个独立 Conda 环境（R / PY / PY2）。环境文件统一放在仓库根目录的 `envs/`。

## 1) 创建环境（在仓库根目录运行）

```bash
conda env create -f envs/R_environment.yml
conda env create -f envs/PY_environment.yml
conda env create -f envs/PY2_environment.yml
```

## 2) prefix 说明（已统一）

这些 yml 文件末尾包含：
- `prefix: ./envs/conda/R`
- `prefix: ./envs/conda/PY`
- `prefix: ./envs/conda/PY2`

这会把环境安装到仓库目录下，方便移动/复制项目。

## 3) 进一步说明

更完整的 Tool-runner 运行指南请看：
- `Tool-runner/SETUP_GUIDE.md`
- `Tool-runner/README.md`

