#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate pathway enrichment results for each clustering method
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import pandas as pd
import gseapy as gp


def generate_pathways(adata_path, output_dir, sample_id, gene_sets="KEGG_2021_Human"):
    """
    Generate pathway enrichment for all aligned domain columns
    
    Args:
        adata_path: Path to aligned h5ad file
        output_dir: Output directory
        sample_id: Sample identifier
        gene_sets: Gene set database for enrichment
    """
    print(f"[Pathways] Starting pathway enrichment for {sample_id}...")
    
    # Load data
    print(f"[Pathways] Loading aligned data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Find all aligned columns
    aligned_cols = [c for c in adata.obs.columns if c.endswith('_domain_aligned')]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for col in aligned_cols:
        base = col.replace("_aligned", "")
        method_name = base.replace("_domain", "")
        key = f"de_genes_{base}"
        
        if key not in adata.uns:
            print(f"[Pathways] Skipping {method_name}: DEG results not found")
            continue
        
        print(f"[Pathways] Processing {method_name}...")
        
        # Build domain rankings
        domain_rankings = {}
        cats = list(adata.obs[col].astype('category').cat.categories)
        for domain in cats:
            try:
                names = adata.uns[key]["names"][str(domain)]
                lfc = adata.uns[key]["logfoldchanges"][str(domain)]
                domain_rankings[domain] = pd.DataFrame({"Gene": names, "Score": lfc})
            except:
                continue
        
        # Run enrichment for each domain
        method_results = []
        for domain, rnk in domain_rankings.items():
            try:
                gsea_res = gp.prerank(
                    rnk=rnk,
                    gene_sets=gene_sets,
                    permutation_num=500,
                    min_size=5,
                    max_size=500,
                    seed=42,
                    no_plot=True
                )
                df = gsea_res.res2d.copy()
                df["Domain"] = domain
                df["abs_NES"] = df["NES"].abs()
                df = df.sort_values("abs_NES", ascending=False).head(20)
                
                keep = [c for c in ["Term","NES","NOM p-val","Lead_genes","Domain"] if c in df.columns]
                method_results.append(df[keep])
            except Exception as e:
                print(f"[Pathways] Warning: Failed for {method_name} domain {domain}: {e}")
                continue
        
        if method_results:
            method_df = pd.concat(method_results, ignore_index=True)
            out_path = os.path.join(output_dir, f"{method_name}_domain_{sample_id}_PATHWAY.csv")
            method_df.to_csv(out_path, index=False)
            print(f"[Pathways] Saved: {out_path}")
    
    print(f"[Pathways] Completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate pathway enrichment for clustering methods')
    parser.add_argument('--adata_path', required=True, help='Path to aligned h5ad file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sample_id', required=True, help='Sample identifier')
    parser.add_argument('--gene_sets', default="KEGG_2021_Human", help='Gene set database')
    
    args = parser.parse_args()
    
    generate_pathways(
        adata_path=args.adata_path,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        gene_sets=args.gene_sets
    )


if __name__ == '__main__':
    main()


