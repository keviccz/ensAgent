#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEDR Clustering Tool
Environment: PY
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import pandas as pd
import torch
from sklearn.decomposition import PCA
import SEDR
import SEDR.SEDR_module as sm
import torch.nn.functional as F


def fix_sedr_processor():
    """Fix SEDR processor function bug"""
    def _processor_func(name):
        if name is None:
            return lambda t: t
        key = str(name).lower()
        if key in ('none', 'identity', 'id'):
            return lambda t: t
        if key in ('relu', 'relu6'):
            return F.relu
        if key == 'tanh':
            return torch.tanh
        if key == 'sigmoid':
            return torch.sigmoid
        return lambda t: t
    
    sm.processor_func = _processor_func
    sm.x = 'relu'


def run_sedr(data_path, sample_id, output_dir, n_clusters=7, random_seed=2023):
    """
    Run SEDR clustering
    
    Args:
        data_path: Path to Visium data directory
        sample_id: Sample identifier
        output_dir: Output directory
        n_clusters: Number of clusters
        random_seed: Random seed
    """
    print(f"[SEDR] Starting clustering for {sample_id}...")
    
    # Set random seed
    SEDR.fix_seed(random_seed)
    fix_sedr_processor()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SEDR] Using device: {device}")
    
    # Load data
    print(f"[SEDR] Loading data from {data_path}...")
    adata = sc.read_visium(data_path)
    adata.var_names_make_unique()
    
    # Preprocessing
    print(f"[SEDR] Preprocessing...")
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    
    # Build graph
    print(f"[SEDR] Building neighborhood graph...")
    graph_dict = SEDR.graph_construction(adata, 12)
    
    # Train SEDR
    print(f"[SEDR] Training model...")
    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    sedr_net.train_with_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat
    
    # Clustering
    print(f"[SEDR] Performing clustering (k={n_clusters})...")
    SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"SEDR_{sample_id}_domain.csv")
    
    pd.DataFrame({
        "spot_id": adata.obs_names,
        "SEDR_domain": adata.obs["SEDR"].astype("Int64")
    }).to_csv(output_file, index=False)
    
    print(f"[SEDR] Results saved to {output_file}")
    print(f"[SEDR] Completed successfully!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='SEDR Spatial Clustering Tool')
    parser.add_argument('--data_path', required=True, help='Path to Visium data directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters (default: 7)')
    parser.add_argument('--random_seed', type=int, default=2023, help='Random seed (default: 2023)')
    
    args = parser.parse_args()
    
    # Setup R environment for mclust
    os.environ['LC_ALL'] = 'C'
    os.environ['LANG'] = 'C'
    
    run_sedr(
        data_path=args.data_path,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


