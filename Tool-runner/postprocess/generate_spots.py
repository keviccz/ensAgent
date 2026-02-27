#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate spot files with spatial coordinates and domain labels
"""

import argparse
import os
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


def generate_spots(adata_path, output_dir, sample_id):
    """
    Generate spot files for all aligned domain columns
    
    Args:
        adata_path: Path to aligned h5ad file
        output_dir: Output directory
        sample_id: Sample identifier
    """
    print(f"[Spots] Starting spot file generation for {sample_id}...")
    
    # Load data
    print(f"[Spots] Loading aligned data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Prepare common metrics
    if 'total_counts' not in adata.obs:
        if 'count' in adata.layers:
            m = adata.layers['count']
            adata.obs['total_counts'] = m.sum(axis=1).A1 if issparse(m) else m.sum(axis=1)
        else:
            X = adata.X
            adata.obs['total_counts'] = X.sum(axis=1).A1 if issparse(X) else X.sum(axis=1)
    
    if 'n_genes' not in adata.obs:
        if 'count' in adata.layers:
            m = adata.layers['count']
            adata.obs['n_genes'] = m.getnnz(axis=1) if issparse(m) else (m > 0).sum(axis=1)
        else:
            X = adata.X
            adata.obs['n_genes'] = X.getnnz(axis=1) if issparse(X) else (X > 0).sum(axis=1)
    
    # Find all aligned columns
    aligned_cols = [c for c in adata.obs.columns if c.endswith('_domain_aligned')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for col in aligned_cols:
        base = col.replace("_aligned", "")
        method_name = base.replace("_domain", "")
        
        print(f"[Spots] Processing {method_name}...")
        
        df = pd.DataFrame({
            'x': adata.obsm['spatial'][:, 0],
            'y': adata.obsm['spatial'][:, 1],
            'in_tissue': adata.obs.get('in_tissue', 1).astype(bool).values,
            'n_genes': adata.obs['n_genes'].values,
            'spatial_domain': adata.obs[col].astype('Int64').values,
        }, index=adata.obs_names)
        
        # Save
        out_path = os.path.join(output_dir, f"{method_name}_domain_{sample_id}_spot.csv")
        df.to_csv(out_path, index=True)
        print(f"[Spots] Saved: {out_path}")
    
    print(f"[Spots] Completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate spot files for clustering methods')
    parser.add_argument('--adata_path', required=True, help='Path to aligned h5ad file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    
    args = parser.parse_args()
    
    generate_spots(
        adata_path=args.adata_path,
        output_dir=args.output_dir,
        sample_id=args.sample_id
    )


if __name__ == '__main__':
    main()


