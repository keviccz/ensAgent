#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain Label Alignment Tool
Aligns domain labels from different clustering methods to a reference
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from collections import defaultdict
from typing import Optional


def align_to_layer_guess_by_overlap(
    adata: AnnData,
    reference_key: str = "layer_guess",
    target_keys: list = None,
    make_numeric_ref_key: str = "layer_guess_id",
    verbose: bool = True,
    enable_flip_check: bool = True,
    flip_corr_threshold: float = 0.55,
    enable_mean_order_fallback: bool = True,
    low_corr_threshold: float = 0.30,
    wm_as: int = 7,
    flip_only_keys: Optional[set] = None,
) -> AnnData:
    """
    Align domain labels to reference using overlap-based matching
    
    Args:
        adata: AnnData object with spatial transcriptomics data
        reference_key: Reference annotation column name
        target_keys: List of target columns to align (auto-detect if None)
        make_numeric_ref_key: Name for numeric reference column
        verbose: Print progress
        enable_flip_check: Auto-flip direction
        flip_corr_threshold: Threshold for flipping
        enable_mean_order_fallback: Use mean layer ordering for low correlation
        low_corr_threshold: Threshold for low correlation
        wm_as: Layer number for white matter
        flip_only_keys: Only flip these specific keys
        
    Returns:
        AnnData with aligned labels
    """
    # Find target columns
    if target_keys is None:
        domain_cols = [c for c in adata.obs.columns if c.endswith("_domain")]
        target_keys = [c for c in domain_cols if c != reference_key]
    if len(target_keys) == 0:
        if verbose:
            print("No target columns found for alignment.")
        return adata

    # Map reference to numeric
    ref_str = adata.obs[reference_key].astype(str).str.strip()
    ref_classes = sorted(pd.unique(ref_str))
    ref_map_str2id = {cl: i + 1 for i, cl in enumerate(ref_classes)}
    if make_numeric_ref_key:
        adata.obs[make_numeric_ref_key] = ref_str.map(ref_map_str2id).astype("Int64")
    if verbose:
        print(f"Reference classes (alphabetical→ID): {ref_map_str2id}")

    # Extract layer numbers from reference
    def extract_layer(layer_str):
        if pd.isna(layer_str): return np.nan
        m = re.search(r'(?:layer|L)[\s_]*?(\d+)|(WM)', str(layer_str), re.IGNORECASE)
        return int(m.group(1)) if m and m.group(1) else (wm_as if m and m.group(2) else np.nan)

    adata.obs['_layer_num_temp'] = adata.obs[reference_key].apply(extract_layer).astype(float)
    valid_mask = ~adata.obs['_layer_num_temp'].isna()

    adata.uns.setdefault('spatial_alignment_mapping', {})

    for spatial_domain_key in target_keys:
        # Mode-based layer mapping
        unique_domains = sorted(adata.obs[spatial_domain_key].unique())
        raw_mapping = {}
        for domain in unique_domains:
            domain_mask = (adata.obs[spatial_domain_key] == domain) & valid_mask
            if domain_mask.sum() > 0:
                common_layer = adata.obs.loc[domain_mask, '_layer_num_temp'].mode()[0]
                raw_mapping[domain] = int(common_layer)
            else:
                raw_mapping[domain] = domain

        layer_groups = defaultdict(list)
        for domain, layer_val in raw_mapping.items():
            layer_groups[layer_val].append(domain)

        domain_layer_map = {}
        label_counter = 1
        for layer_val, domains in sorted(layer_groups.items()):
            for domain in sorted(domains):
                domain_layer_map[domain] = label_counter
                label_counter += 1

        out_col = f"{spatial_domain_key}_aligned"
        adata.obs[out_col] = adata.obs[spatial_domain_key].map(domain_layer_map).astype(int)

        # Mean order fallback for low correlation
        if enable_mean_order_fallback and valid_mask.any():
            aligned_vals = adata.obs.loc[valid_mask, out_col].astype(float)
            ref_vals = adata.obs.loc[valid_mask, '_layer_num_temp'].astype(float)
            rho = aligned_vals.corr(ref_vals, method='spearman') if len(aligned_vals) > 2 else np.nan
            if verbose:
                print(f"[{spatial_domain_key}] Spearman correlation: {rho:.3f}" if np.isfinite(rho) else f"[{spatial_domain_key}] Spearman correlation: nan")

            if (rho is not None) and np.isfinite(rho) and (rho < low_corr_threshold):
                mean_layer = {}
                for d in unique_domains:
                    m = (adata.obs[spatial_domain_key] == d) & valid_mask
                    if m.sum() > 0:
                        mean_layer[d] = float(adata.obs.loc[m, '_layer_num_temp'].mean())
                    else:
                        mean_layer[d] = float(np.nanmedian(adata.obs.loc[valid_mask, '_layer_num_temp']))
                order = sorted(unique_domains, key=lambda d: (mean_layer[d], str(d)))
                new_map = {d: i + 1 for i, d in enumerate(order)}
                adata.obs[out_col] = adata.obs[spatial_domain_key].map(new_map).astype(int)
                domain_layer_map = new_map
                if verbose:
                    print(f"[{spatial_domain_key}] Applied mean layer ordering fallback")

                aligned_vals = adata.obs.loc[valid_mask, out_col].astype(float)
                rho = aligned_vals.corr(ref_vals, method='spearman') if len(aligned_vals) > 2 else rho
                if verbose and np.isfinite(rho):
                    print(f"[{spatial_domain_key}] Spearman after fallback: {rho:.3f}")

        # Auto-flip direction
        if enable_flip_check and valid_mask.any():
            if (flip_only_keys is None) or (spatial_domain_key in flip_only_keys):
                aligned_vals = adata.obs.loc[valid_mask, out_col].astype(float)
                ref_vals = adata.obs.loc[valid_mask, '_layer_num_temp'].astype(float)
                if len(aligned_vals) > 2:
                    rho = aligned_vals.corr(ref_vals, method='spearman')
                    if (rho is not None) and np.isfinite(rho) and (rho < -flip_corr_threshold):
                        mn, mx = int(adata.obs[out_col].min()), int(adata.obs[out_col].max())
                        adata.obs[out_col] = mn + mx - adata.obs[out_col]
                        if verbose:
                            print(f"[{spatial_domain_key}] Applied direction flip: i → {mn}+{mx}-i")

        # Validation
        original_domains = set(adata.obs[spatial_domain_key].unique())
        mapped_domains = set(adata.obs[out_col].unique())
        if len(original_domains) != len(mapped_domains):
            lost = original_domains - set(domain_layer_map.keys())
            extra = mapped_domains - set(domain_layer_map.values())
            raise RuntimeError(
                f"{spatial_domain_key}: Domain count mismatch! Original={len(original_domains)}, New={len(mapped_domains)}\n"
                f"Lost domains: {lost}\nExtra domains: {extra}"
            )
        min_label, max_label = min(mapped_domains), max(mapped_domains)
        expected_labels = set(range(min_label, max_label + 1))
        if mapped_domains != expected_labels:
            missing_labels = expected_labels - mapped_domains
            raise RuntimeError(f"{spatial_domain_key}: Non-continuous labels! Missing: {missing_labels}")

        adata.uns['spatial_alignment_mapping'][spatial_domain_key] = domain_layer_map
        if verbose:
            print(f"✅ Validation passed [{spatial_domain_key}]: Labels range {min_label}-{max_label}")

    del adata.obs['_layer_num_temp']
    return adata


def main():
    parser = argparse.ArgumentParser(description='Domain Label Alignment Tool')
    parser.add_argument('--data_path', required=True, help='Path to Visium data directory')
    parser.add_argument('--domain_files', required=True, nargs='+', help='List of domain CSV files')
    parser.add_argument('--reference_col', default='layer_guess', help='Reference column name')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    
    args = parser.parse_args()
    
    print(f"[Alignment] Starting alignment for {args.sample_id}...")
    
    # Load base data
    print(f"[Alignment] Loading data from {args.data_path}...")
    adata = sc.read_visium(args.data_path)
    adata.var_names_make_unique()
    
    # Load reference if available
    metadata_path = os.path.join(args.data_path, 'metadata.tsv')
    if os.path.exists(metadata_path):
        df_meta = pd.read_csv(metadata_path, sep='\t')
        adata.obs[args.reference_col] = df_meta[args.reference_col]
        print(f"[Alignment] Loaded reference from {metadata_path}")
    
    # Load domain files
    for domain_file in args.domain_files:
        if not os.path.exists(domain_file):
            print(f"[Alignment] Warning: {domain_file} not found, skipping")
            continue
            
        method_name = os.path.basename(domain_file).split('_')[0]
        
        # Read CSV and handle index properly
        df = pd.read_csv(domain_file)
        if 'spot_id' in df.columns:
            df = df.set_index('spot_id')
        elif df.columns[0] == 'Unnamed: 0':  # First column is index without name
            df = df.set_index(df.columns[0])
            df.index.name = 'spot_id'
        
        # Find domain column
        domain_cols = [c for c in df.columns if 'domain' in c.lower()]
        if not domain_cols:
            print(f"[Alignment] Warning: No domain column found in {domain_file}, skipping")
            continue
        domain_col = domain_cols[0]
        col_name = f"{method_name}_domain"
        
        common = adata.obs_names.intersection(df.index)
        if len(common) == 0:
            print(f"[Alignment] Warning: No overlap for {method_name}, skipping")
            continue
            
        adata._inplace_subset_obs(common)
        adata.obs[col_name] = pd.Categorical(df.loc[adata.obs_names, domain_col].astype(str))
        print(f"[Alignment] Loaded {method_name}: {len(common)} spots")
    
    # Remove missing values
    adata = adata[~adata.obs[args.reference_col].isin(['']) & adata.obs[args.reference_col].notna()].copy()
    
    # Run alignment
    print(f"[Alignment] Running alignment algorithm...")
    adata = align_to_layer_guess_by_overlap(
        adata,
        reference_key=args.reference_col,
        target_keys=None,
        verbose=True
    )
    
    # Save aligned h5ad
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.sample_id}_aligned.h5ad")
    adata.write(output_file)
    
    print(f"[Alignment] Aligned data saved to {output_file}")
    print(f"[Alignment] Completed successfully!")
    
    return output_file


if __name__ == '__main__':
    main()


