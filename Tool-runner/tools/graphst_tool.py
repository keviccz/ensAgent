#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphST Clustering Tool
Environment: PY
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from GraphST import GraphST


def run_graphst(data_path, sample_id, output_dir, n_clusters=7, random_seed=2023):
    """
    Run GraphST clustering
    
    Args:
        data_path: Path to Visium data directory
        sample_id: Sample identifier
        output_dir: Output directory
        n_clusters: Number of clusters
        random_seed: Random seed
    """
    print(f"[GraphST] Starting clustering for {sample_id}...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GraphST] Using device: {device}")
    
    # Load data
    print(f"[GraphST] Loading data from {data_path}...")
    adata = sc.read_visium(data_path)
    adata.var_names_make_unique()
    
    # Preprocessing
    print(f"[GraphST] Preprocessing...")
    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    
    adata_X = PCA(n_components=200, random_state=random_seed).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X
    
    # Run GraphST
    print(f"[GraphST] Training model...")
    model = GraphST.GraphST(adata, device=device, random_seed=random_seed)
    adata = model.train()
    
    # Clustering
    print(f"[GraphST] Performing clustering (k={n_clusters})...")
    from GraphST.utils import clustering
    radius = 50
    tool = 'mclust'
    
    clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"GraphST_{sample_id}_domain.csv")
    
    pd.DataFrame({
        "spot_id": adata.obs_names,
        "GraphST_domain": adata.obs["domain"].astype("Int64")
    }).to_csv(output_file, index=False)
    
    print(f"[GraphST] Results saved to {output_file}")
    print(f"[GraphST] Completed successfully!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='GraphST Spatial Clustering Tool')
    parser.add_argument('--data_path', required=True, help='Path to Visium data directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters (default: 7)')
    parser.add_argument('--random_seed', type=int, default=2023, help='Random seed (default: 2023)')
    
    args = parser.parse_args()
    
    run_graphst(
        data_path=args.data_path,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


