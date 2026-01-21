# Examples

This directory contains example scripts and data for running the Tool-Runner Agent.

## Quick Start

### Windows

```cmd
cd examples
run_demo.bat
```

### Linux/Mac

```bash
cd examples
chmod +x run_demo.sh
./run_demo.sh
```

## Before Running

1. **Create conda environments** (from project root):
   ```bash
   conda env create -f envs/R_environment.yml
   conda env create -f envs/PY_environment.yml
   conda env create -f envs/PY2_environment.yml
   ```

2. **Update data path** in `configs/DLPFC_151507.yaml`:
   ```yaml
   data_path: "D:/path/to/your/data/151507"  # Change this!
   ```

3. **Verify data directory** contains:
   - `filtered_feature_bc_matrix.h5`
   - `metadata.tsv` (optional)
   - `spatial/` folder with images and positions

## Example Command

```bash
# From project root
python orchestrator.py --config configs/DLPFC_151507.yaml
```

## Expected Output

```
output/DLPFC_151507/
├── domains/           # 8 clustering result files
├── spot/              # 8 spot data files
├── PICTURES/          # 8 visualization images
├── DEGs/              # 8 DEG analysis files
├── PATHWAY/           # 8 pathway enrichment files
├── DLPFC_151507_aligned.h5ad
└── tool_runner_report.json
```

## Troubleshooting

If you encounter errors:

1. Check `output/DLPFC_151507/tool_runner_report.json`
2. Verify all conda environments are created
3. Ensure data paths are correct
4. Check that you have sufficient disk space (>5GB recommended)

For more help, see the main [README.md](../README.md)

