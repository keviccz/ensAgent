#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tool-Runner Agent Orchestrator
Coordinates execution of spatial clustering tools, alignment, and downstream analysis
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed.")
    print("Please install it with: pip install pyyaml")
    sys.exit(1)


# Method -> environment label mapping
_METHOD_ENV: dict[str, str] = {
    "IRIS": "R",
    "BASS": "R",
    "DR-SC": "R",
    "BayesSpace": "R",
    "SEDR": "PY",
    "GraphST": "PY",
    "STAGATE": "PY",
    "stLearn": "PY2",
}


class ToolRunnerAgent:
    """Main orchestrator for spatial clustering pipeline."""

    def __init__(self, config_path=None, **kwargs):
        self.config = self._load_config(config_path, kwargs)
        self.results = {
            "sample_id": self.config["sample_id"],
            "status": "initialized",
            "methods_executed": [],
            "methods_failed": [],
            "output_files": {},
            "start_time": datetime.now().isoformat(),
        }

    def _load_config(self, config_path, kwargs):
        config = {}

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

        for key, value in kwargs.items():
            if value is not None:
                config[key] = value

        config.setdefault("n_clusters", 7)
        config.setdefault("random_seed", 2023)
        config.setdefault("methods", [
            "IRIS", "BASS", "DR-SC", "BayesSpace",
            "SEDR", "GraphST", "STAGATE", "stLearn",
        ])
        config.setdefault("min_success", 5)
        config.setdefault("timeout", 3600)

        # conda/mamba executable (default: mamba)
        config.setdefault("conda_exe", "mamba")
        # per-label environment names (can be overridden from CLI or YAML)
        config.setdefault("env_r", "R")
        config.setdefault("env_py", "PY")
        config.setdefault("env_py2", "PY2")

        required = ["data_path", "sample_id", "output_dir"]
        missing = [k for k in required if k not in config or not config[k]]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        if not os.path.exists(config["data_path"]):
            raise ValueError(f"Data path does not exist: {config['data_path']}")

        return config

    def _env_name_for(self, method: str) -> str | None:
        """Resolve the actual conda/mamba env name for *method*."""
        label = _METHOD_ENV.get(method)
        if label is None:
            return None
        key = f"env_{label.lower()}"
        return self.config.get(key, label)

    def _run_tool(self, method):
        """Run a single clustering tool via ``mamba run``."""
        print(f"\n{'=' * 60}")
        print(f"Running {method}...")
        print(f"{'=' * 60}")

        tool_dir = Path(__file__).parent / "tools"
        data_path = str(Path(self.config["data_path"]).absolute())
        sample_id = self.config["sample_id"]
        output_dir = Path(self.config["output_dir"]) / "domains"
        n_clusters = self.config["n_clusters"]
        random_seed = self.config["random_seed"]
        timeout = self.config.get("timeout", 3600)
        conda_exe = self.config.get("conda_exe", "mamba")

        output_dir.mkdir(parents=True, exist_ok=True)

        env_name = self._env_name_for(method)

        if method in ("IRIS", "BASS", "DR-SC", "BayesSpace"):
            script = str((tool_dir / f"{method.lower().replace('-', '')}_tool.R").absolute())
            tool_cmd = [
                "Rscript", script,
                "--data_path", data_path,
                "--sample_id", sample_id,
                "--output_dir", str(output_dir),
                "--n_clusters", str(n_clusters),
                "--random_seed", str(random_seed),
            ]
        else:
            script = str((tool_dir / f"{method.lower()}_tool.py").absolute())
            tool_cmd = [
                "python", script,
                "--data_path", data_path,
                "--sample_id", sample_id,
                "--output_dir", str(output_dir),
                "--n_clusters", str(n_clusters),
                "--random_seed", str(random_seed),
            ]

        # Use `mamba run` / `conda run` to execute inside the target environment.
        # This avoids the need for shell hooks (`conda activate`) in subprocesses.
        if env_name:
            cmd = [conda_exe, "run", "-n", env_name, "--no-banner", *tool_cmd]
        else:
            cmd = tool_cmd

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                print(f"✓ {method} completed successfully")
                self.results["methods_executed"].append(method)
                return True
            else:
                print(f"✗ {method} failed with return code {result.returncode}")
                if result.stdout:
                    print(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")
                self.results["methods_failed"].append(method)
                return False

        except subprocess.TimeoutExpired:
            print(f"✗ {method} timed out after {timeout} seconds")
            self.results["methods_failed"].append(method)
            return False
        except Exception as e:
            print(f"✗ {method} failed with exception: {e}")
            import traceback
            print(traceback.format_exc())
            self.results["methods_failed"].append(method)
            return False
    
    def run_clustering_tools(self):
        """Run all clustering tools"""
        print(f"\n{'#'*60}")
        print(f"# PHASE 1: Clustering Tools Execution")
        print(f"# Sample: {self.config['sample_id']}")
        print(f"# Methods: {', '.join(self.config['methods'])}")
        print(f"{'#'*60}\n")
        
        for method in self.config['methods']:
            self._run_tool(method)
        
        n_success = len(self.results['methods_executed'])
        n_failed = len(self.results['methods_failed'])
        
        print(f"\n{'='*60}")
        print(f"Clustering Summary:")
        print(f"  Successful: {n_success}/{len(self.config['methods'])}")
        print(f"  Failed: {n_failed}")
        print(f"{'='*60}\n")
        
        if n_success < self.config['min_success']:
            raise RuntimeError(f"Only {n_success} methods succeeded, minimum {self.config['min_success']} required")
        
        return n_success >= self.config['min_success']
    
    def run_alignment(self):
        """Run label alignment."""
        print(f"\n{'#' * 60}")
        print(f"# PHASE 2: Domain Label Alignment")
        print(f"{'#' * 60}\n")

        align_script = str(Path(__file__).parent / "postprocess" / "align_labels.py")
        data_path = self.config["data_path"]
        sample_id = self.config["sample_id"]
        output_dir = Path(self.config["output_dir"])

        domain_dir = output_dir / "domains"
        domain_files = list(domain_dir.glob(f"*_{sample_id}_domain.csv"))

        if not domain_files:
            raise RuntimeError("No domain files found for alignment")

        cmd = [
            sys.executable, align_script,
            "--data_path", str(data_path),
            "--domain_files", *[str(f) for f in domain_files],
            "--output_dir", str(output_dir),
            "--sample_id", sample_id,
        ]
        if "reference_col" in self.config:
            cmd += ["--reference_col", self.config["reference_col"]]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✓ Alignment completed successfully")
            aligned_file = output_dir / f"{sample_id}_aligned.h5ad"
            self.results["output_files"]["aligned_data"] = str(aligned_file)
            return True
        else:
            print("✗ Alignment failed")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            return False

    def run_downstream_analysis(self):
        """Run downstream analyses (DEGs, pathways, spots, pictures)."""
        print(f"\n{'#' * 60}")
        print(f"# PHASE 3: Downstream Analysis")
        print(f"{'#' * 60}\n")

        postprocess_dir = Path(__file__).parent / "postprocess"
        output_dir = Path(self.config["output_dir"])
        sample_id = self.config["sample_id"]
        aligned_file = output_dir / f"{sample_id}_aligned.h5ad"

        if not aligned_file.exists():
            raise RuntimeError(f"Aligned data file not found: {aligned_file}")

        analyses = [
            ("DEGs", "generate_degs.py", output_dir / "DEGs"),
            ("Spots", "generate_spots.py", output_dir / "spot"),
            ("Pathways", "generate_pathways.py", output_dir / "PATHWAY"),
            ("Pictures", "generate_pictures.py", output_dir / "PICTURES"),
        ]

        for name, script, out_dir in analyses:
            print(f"\nGenerating {name}...")
            script_path = str(postprocess_dir / script)
            cmd = [
                sys.executable, script_path,
                "--adata_path", str(aligned_file),
                "--output_dir", str(out_dir),
                "--sample_id", sample_id,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ {name} generated successfully")
                self.results["output_files"][name.lower()] = str(out_dir)
            else:
                print(f"✗ {name} generation failed")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")

        return True
    
    def generate_report(self):
        """Generate execution report"""
        self.results['end_time'] = datetime.now().isoformat()
        self.results['status'] = 'completed'
        
        report_path = Path(self.config['output_dir']) / "tool_runner_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'#'*60}")
        print(f"# EXECUTION REPORT")
        print(f"{'#'*60}")
        print(f"Sample: {self.results['sample_id']}")
        print(f"Methods executed: {len(self.results['methods_executed'])}/{len(self.config['methods'])}")
        print(f"  Successful: {', '.join(self.results['methods_executed'])}")
        if self.results['methods_failed']:
            print(f"  Failed: {', '.join(self.results['methods_failed'])}")
        print(f"\nOutput directory: {self.config['output_dir']}")
        print(f"Report saved to: {report_path}")
        print(f"{'#'*60}\n")
    
    def run(self):
        """Run the complete pipeline"""
        try:
            # Phase 1: Clustering
            self.run_clustering_tools()
            
            # Phase 2: Alignment
            self.run_alignment()
            
            # Phase 3: Downstream analysis
            self.run_downstream_analysis()
            
            # Generate report
            self.generate_report()
            
            print("\n✓ Pipeline completed successfully!\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}\n")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.generate_report()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Tool-Runner Agent: Spatial Transcriptomics Clustering Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --config configs/DLPFC_151507.yaml
  python orchestrator.py --data_path /path/to/data --sample_id DLPFC_151507 --output_dir ./output
        """,
    )

    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--data_path", help="Path to Visium data directory")
    parser.add_argument("--sample_id", help="Sample identifier")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--n_clusters", type=int, help="Number of clusters (default: 7)")
    parser.add_argument("--random_seed", type=int, help="Random seed (default: 2023)")
    parser.add_argument("--methods", nargs="+", help="Methods to run")
    parser.add_argument("--reference_col", help="Reference column name for alignment")
    parser.add_argument("--conda_exe", default=None, help='Conda/mamba executable (default: "mamba")')
    parser.add_argument("--env_r", default=None, help='R environment name (default: "R")')
    parser.add_argument("--env_py", default=None, help='PY environment name (default: "PY")')
    parser.add_argument("--env_py2", default=None, help='PY2 environment name (default: "PY2")')

    args = parser.parse_args()

    agent = ToolRunnerAgent(
        config_path=args.config,
        data_path=args.data_path,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed,
        methods=args.methods,
        reference_col=args.reference_col,
        conda_exe=args.conda_exe,
        env_r=args.env_r,
        env_py=args.env_py,
        env_py2=args.env_py2,
    )

    success = agent.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


