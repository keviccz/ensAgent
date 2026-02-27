#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stLearn Clustering Tool
Environment: PY2
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import stlearn as st
import pandas as pd


def run_stlearn(data_path, sample_id, output_dir, n_clusters=7, random_seed=2023):
    """
    Run stLearn stSME clustering
    
    Args:
        data_path: Path to Visium data directory
        sample_id: Sample identifier
        output_dir: Output directory
        n_clusters: Number of clusters
        random_seed: Random seed
    """
    print(f"[stLearn] Starting clustering for {sample_id}...")
    
    # Load data
    print(f"[stLearn] Loading data from {data_path}...")
    spatial_dir = os.path.join(data_path, "spatial")
    adata = sc.read_visium(
        path=data_path,
        count_file="filtered_feature_bc_matrix.h5",
        source_image_path=spatial_dir
    )
    adata.var_names_make_unique()
    
    # Preprocessing
    print(f"[stLearn] Preprocessing...")
    st.pp.filter_genes(adata, min_cells=3)
    st.pp.normalize_total(adata)
    st.pp.log1p(adata)
    
    # SME normalization
    print(f"[stLearn] Running SME normalization...")
    st.pp.tiling(adata, out_path=os.path.join(output_dir, 'tiles'))
    st.pp.extract_feature(adata)
    st.em.run_pca(adata, n_comps=50, random_state=random_seed)
    
    # stSME
    print(f"[stLearn] Running stSME...")
    st.spatial.SME.SME_normalize(adata, use_data="raw", weights="weights_matrix_all")
    
    # Clustering
    print(f"[stLearn] Performing clustering (k={n_clusters})...")
    st.pp.scale(adata)
    st.em.run_pca(adata, n_comps=50, random_state=random_seed)
    st.tl.clustering.kmeans(adata, n_clusters=n_clusters, use_data="X_pca", key_added="stLearn_domain")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"stLearn_{sample_id}_domain.csv")
    
    pd.DataFrame({
        "spot_id": adata.obs_names,
        "stLearn_domain": adata.obs["stLearn_domain"].astype("Int64")
    }).to_csv(output_file, index=False)
    
    print(f"[stLearn] Results saved to {output_file}")
    print(f"[stLearn] Completed successfully!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='stLearn Spatial Clustering Tool')
    parser.add_argument('--data_path', required=True, help='Path to Visium data directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters (default: 7)')
    parser.add_argument('--random_seed', type=int, default=2023, help='Random seed (default: 2023)')
    
    args = parser.parse_args()
    
    run_stlearn(
        data_path=args.data_path,
        sample_id=args.sample_id,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


