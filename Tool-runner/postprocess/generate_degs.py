#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate DEGs (Differentially Expressed Genes) for each clustering method
"""

import argparse
import os
import scanpy as sc
import pandas as pd


def generate_degs(adata_path, output_dir, sample_id, min_pct=0.1):
    """
    Generate DEGs for all aligned domain columns
    
    Args:
        adata_path: Path to aligned h5ad file
        output_dir: Output directory
        sample_id: Sample identifier
        min_pct: Minimum percentage of cells for gene filtering
    """
    print(f"[DEGs] Starting DEG analysis for {sample_id}...")
    
    # Load data
    print(f"[DEGs] Loading aligned data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Preprocessing for DEG analysis
    print(f"[DEGs] Preprocessing...")
    if 'count' in adata.layers:
        sc.pp.normalize_total(adata, target_sum=1e4, layer='count')
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
    
    sc.pp.filter_genes(adata, min_cells=int(adata.n_obs * min_pct))
    print(f"[DEGs] Genes after filtering: {adata.n_vars}")
    
    # Find all aligned columns
    aligned_cols = [c for c in adata.obs.columns if c.endswith('_domain_aligned')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for col in aligned_cols:
        base = col.replace("_aligned", "")
        method_name = base.replace("_domain", "")
        key = f"de_genes_{base}"
        
        print(f"[DEGs] Processing {method_name}...")
        
        # Ensure categorical
        adata.obs[col] = adata.obs[col].astype("category")
        
        # Differential expression analysis
        sc.tl.rank_genes_groups(
            adata,
            groupby=col,
            layer='count' if 'count' in adata.layers else None,
            method='wilcoxon',
            use_raw=False,
            key_added=key
        )
        
        # Extract and filter results
        deg_df = sc.get.rank_genes_groups_df(adata, group=None, key=key)
        deg_df = deg_df[(deg_df['pvals_adj'] < 0.05) & (deg_df['logfoldchanges'].abs() > 1)]
        deg_df = (
            deg_df.sort_values(['group', 'pvals_adj'])
                  .groupby('group')
                  .head(10)
                  .reset_index(drop=True)
                  .rename(columns={'group': 'domain'})
        )
        
        # Save results
        out_path = os.path.join(output_dir, f"{method_name}_domain_{sample_id}_DEGs.csv")
        deg_df.to_csv(out_path, index=False)
        print(f"[DEGs] Saved: {out_path}")
    
    print(f"[DEGs] Completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate DEGs for clustering methods')
    parser.add_argument('--adata_path', required=True, help='Path to aligned h5ad file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    parser.add_argument('--min_pct', type=float, default=0.1, help='Minimum cell percentage for gene filtering')
    
    args = parser.parse_args()
    
    generate_degs(
        adata_path=args.adata_path,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        min_pct=args.min_pct
    )


if __name__ == '__main__':
    main()


