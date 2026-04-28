# Conda Environment Setup for Tool-Runner

Tool-Runner uses three separate Conda/Mamba environments for method execution:

- `R`: IRIS, BASS, DR-SC, BayesSpace
- `PY`: SEDR, GraphST, STAGATE
- `PY2`: stLearn

The canonical environment files are stored under `envs/`.

## Create Environments

Run these commands from the repository root:

```bash
mamba env create -f envs/R_environment.yml
mamba env create -f envs/PY_environment.yml
mamba env create -f envs/PY2_environment.yml
```

`conda env create` can be used instead of `mamba env create`, but Mamba is recommended for faster solving.

## Environment Prefixes

The YAML files install environments under the repository-local `envs/conda/` directory:

- `envs/conda/R`
- `envs/conda/PY`
- `envs/conda/PY2`

This keeps method environments portable with the project checkout.

## More Information

See the Tool-Runner documentation for method-specific setup and execution details:

- `Tool-runner/SETUP_GUIDE.md`
- `Tool-runner/README.md`
