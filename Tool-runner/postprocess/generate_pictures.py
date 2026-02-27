#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate spatial domain visualization pictures
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc


def generate_pictures(adata_path, output_dir, sample_id):
    """
    Generate spatial plots for all aligned domain columns
    
    Args:
        adata_path: Path to aligned h5ad file
        output_dir: Output directory
        sample_id: Sample identifier
    """
    print(f"[Pictures] Starting visualization generation for {sample_id}...")
    
    # Load data
    print(f"[Pictures] Loading aligned data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Find all aligned columns
    aligned_cols = [c for c in adata.obs.columns if c.endswith('_domain_aligned')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for col in aligned_cols:
        base = col.replace("_aligned", "")
        method_name = base.replace("_domain", "")
        
        print(f"[Pictures] Processing {method_name}...")
        
        # Ensure categorical
        adata.obs[col] = adata.obs[col].astype("category")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        sc.pl.spatial(
            adata,
            img_key="hires" if "hires" in adata.uns.get('spatial', {}).get(list(adata.uns.get('spatial', {}).keys())[0] if adata.uns.get('spatial') else '', {}).get('images', {}) else None,
            color=[col],
            legend_loc="right margin",
            show=False,
            ax=ax
        )
        plt.title(f"{method_name}_{sample_id}", fontsize=14)
        
        # Save
        out_path = os.path.join(output_dir, f"{method_name}_domain_{sample_id}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[Pictures] Saved: {out_path}")
    
    print(f"[Pictures] Completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate spatial visualizations for clustering methods')
    parser.add_argument('--adata_path', required=True, help='Path to aligned h5ad file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    
    args = parser.parse_args()
    
    generate_pictures(
        adata_path=args.adata_path,
        output_dir=args.output_dir,
        sample_id=args.sample_id
    )


if __name__ == '__main__':
    main()


