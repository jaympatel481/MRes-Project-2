import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import BayesianGaussianMixture

def parse_vcf_with_vaf(vcf_file):
    variants, varid_vaf = [], {}
    with open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt_alleles = fields[4].split(',')
            info = fields[8].split(':')
            vals = fields[9].split(':')
            try:
                af = float(vals[info.index('AF')].split(',')[0])
            except Exception:
                af = 0.0
            for alt in alt_alleles:
                vname = f"{chrom}_{pos}_{ref}_{alt}"
                variants.append({'varid': vname, 'chrom': chrom, 'pos': pos, 'vaf': af})
                varid_vaf[vname] = af
    return variants, varid_vaf

def safe_split_vaf(vaf_field):
    if not isinstance(vaf_field, str):
        if pd.isna(vaf_field):
            return []
        else:
            vaf_field = str(vaf_field)
    return [float(v) for v in vaf_field.split(';') if v]

def load_blocks_csv(blocks_file):
    blocks_df = pd.read_csv(blocks_file)
    blocks = []
    for col in blocks_df.columns:
        col_variants = [v for v in blocks_df[col].dropna().astype(str) if v.strip()]
        blocks.append(col_variants)
    return blocks

def load_sublineage_variants(sublineage_file):
    with open(sublineage_file) as f:
        return set(line.strip() for line in f if line.strip())

def assign_clusters_dpmms(variants, n_init=100, random_state=4):
    vaf_array = np.array([v['vaf'] for v in variants]).reshape(-1, 1)
    bgm = BayesianGaussianMixture(n_components=4, weight_concentration_prior_type='dirichlet_process',
                                  n_init=n_init, random_state=random_state)
    clusters = bgm.fit_predict(vaf_array)
    return clusters

def split_block_by_cluster(block, varid_cluster, varid_pos):
    """
    Given a block (list of variant ids), and the cluster dict and positions, return a list of lists,
    where each inner list contains only contiguous variants with the same cluster label.
    """
    # Sort block by position
    block_sorted = sorted(block, key=lambda v: varid_pos.get(v, 0))
    subblocks = []
    current_subblock = []
    current_cluster = None
    for v in block_sorted:
        v_c = varid_cluster.get(v)
        if v_c is None:
            continue
        if current_cluster is None or v_c == current_cluster:
            current_subblock.append(v)
            current_cluster = v_c
        else:
            if current_subblock:
                subblocks.append((current_cluster, current_subblock))
            current_subblock = [v]
            current_cluster = v_c
    if current_subblock:
        subblocks.append((current_cluster, current_subblock))
    return subblocks

def blockwise_analysis(clusters, variants, varid_vaf, blocks, sublineage_variants, out_prefix):
    varid_cluster = {v['varid']: clusters[i] for i, v in enumerate(variants)}
    varid_pos = {v['varid']: v['pos'] for v in variants}
    cluster_labels = np.unique(clusters)
    records = []
    var_df = pd.DataFrame([{**v, 'cluster': clusters[i]} for i, v in enumerate(variants)])
    # --- NEW: split blocks at cluster transitions ---
    subblocks = []
    for block in blocks:
        split_blocks = split_block_by_cluster(block, varid_cluster, varid_pos)
        # each is (cluster_label, [variant_ids])
        subblocks.extend(split_blocks)

    # For cluster summaries: find which subblocks belong to each cluster
    for c in cluster_labels:
        cluster_varids = [v['varid'] for v in variants if varid_cluster[v['varid']] == c]
        # Only subblocks fully in this cluster
        cluster_subblocks = [blk for cl, blk in subblocks if cl == c and blk]
        # List all variants in these subblocks
        cluster_block_variants = set([v for blk in cluster_subblocks for v in blk])
        blocks_with_sublineage = []
        sublineage_in_blocks = []
        non_sublineage_in_sub_blocks = []
        all_in_sub_blocks = set()
        for blk in cluster_subblocks:
            blk_subs = [v for v in blk if v in sublineage_variants]
            if blk_subs:
                blocks_with_sublineage.append(blk)
                sublineage_in_blocks.extend(blk_subs)
                non_sublineage_in_sub_blocks.extend([v for v in blk if v not in sublineage_variants])
                all_in_sub_blocks |= set(blk)
        cluster_vars_set = set(cluster_varids)
        block_sub_overlap = set()
        for blk in blocks_with_sublineage:
            block_sub_overlap |= set(blk)
        cluster_vars_not_in_any_sub_block = list(cluster_vars_set - block_sub_overlap)
        #  cluster_vars_not_in_any_sub_block = list(set(cluster_vars_not_in_any_sub_block) - set(sublineage_variants))
        vafs_in_sub_blocks, vafs_in_no_sub_blocks = [], []
        for blk in cluster_subblocks:
            vafs = [varid_vaf[v] for v in blk if v in varid_vaf]
            if any(v in sublineage_variants for v in blk):
                vafs_in_sub_blocks.extend(vafs)
            else:
                vafs_in_no_sub_blocks.extend(vafs)
        records.append({
            'cluster_label': c,
            'variants_in_cluster': cluster_varids,
            'blocks_in_cluster': len(cluster_subblocks),
            'num_sublineage_blocks': len(blocks_with_sublineage),
            'sublineage_variants_in_blocks': list(set(sublineage_in_blocks)),
            'non_sublineage_variants_in_sublineage_blocks': list(set(non_sublineage_in_sub_blocks)),
            'variants_not_in_sublineage_blocks': cluster_vars_not_in_any_sub_block,
            'vafs_in_sublineage_blocks': vafs_in_sub_blocks,
            'vafs_in_no_sub_blocks': vafs_in_no_sub_blocks,
        })
        # Per-cluster CSVs
        pd.DataFrame({'variant': cluster_varids, 'vaf': [varid_vaf[v] for v in cluster_varids]}).to_csv(
            f"{out_prefix}_cluster{c}_variants.csv", index=False)
        pd.DataFrame({'block': list(range(len(cluster_subblocks))), 'variants': cluster_subblocks}).to_csv(
            f"{out_prefix}_cluster{c}_blocks.csv", index=False)
        pd.DataFrame({'variant': list(set(sublineage_in_blocks))}).to_csv(
            f"{out_prefix}_cluster{c}_sublineage_in_blocks.csv", index=False)
        pd.DataFrame({'variant': list(set(non_sublineage_in_sub_blocks))}).to_csv(
            f"{out_prefix}_cluster{c}_non_sublineage_in_blocks.csv", index=False)

    cluster_table = []
    for rec in records:
        cluster_table.append({
            'cluster_label': rec['cluster_label'],
            'n_variants': len(rec['variants_in_cluster']),
            'n_blocks': rec['blocks_in_cluster'],
            'num_sublineage_blocks': rec['num_sublineage_blocks'],
            'sublineage_vars': ';'.join(rec['sublineage_variants_in_blocks']),
            'non_sublineage_in_sublineage_blocks': ';'.join(rec['non_sublineage_variants_in_sublineage_blocks']),
            'variants_not_in_sublineage_blocks': ';'.join(rec['variants_not_in_sublineage_blocks']),
            'vafs_in_sublineage_blocks': ';'.join(map(str, rec['vafs_in_sublineage_blocks'])),
            'vafs_in_no_sub_blocks': ';'.join(map(str, rec['vafs_in_no_sub_blocks'])),
        })
    cluster_df = pd.DataFrame(cluster_table)
    cluster_df.to_csv(f"{out_prefix}_cluster_analysis.csv", index=False)
    print(f"Per-cluster results written under prefix {out_prefix}_cluster*.csv and summary to {out_prefix}_cluster_analysis.csv.")

def plot_vaf_clusters(var_df, sublineage_variants, blocks, out_prefix):
    phasing_context = []
    block_membership = {}
    for block in blocks:
        for v in block:
            block_membership[v] = block_membership.get(v, []) + [tuple(block)]
    for idx, row in var_df.iterrows():
        varid = row['varid']
        if varid in sublineage_variants:
            context = 'Sublineage'
        else:
            coexist = any(any(x in sublineage_variants for x in block) for block in block_membership.get(varid, []))
            if coexist:
                context = 'PhasedWithSublineage'
            else:
                context = 'NotPhasedWithSublineage'
        phasing_context.append(context)
    var_df['phasing_context'] = phasing_context

    plt.figure(figsize=(11,5))
    palette = {'Sublineage':'#D62728', 'PhasedWithSublineage': '#1f77b4', 'NotPhasedWithSublineage':'#7F7F7F'}
    sns.histplot(var_df, x='vaf', hue='cluster', bins=30, element='step', linewidth=1.5, palette='tab10',
                 multiple='layer', alpha=0.5)
    for ctx, col, marker in zip(['Sublineage', 'PhasedWithSublineage', 'NotPhasedWithSublineage'],
                                ['#D62728', '#1f77b4', '#7F7F7F'],
                                ['o', 's', '^']):
        ctx_vars = var_df[var_df['phasing_context'] == ctx]
        plt.scatter(ctx_vars['vaf'], [0.5]*len(ctx_vars), alpha=0.7, color=col, marker=marker, label=ctx, s=80)
    plt.xlabel('Variant Allele Frequency')
    plt.ylabel('Variant Histogram + Annotation')
    plt.title('VAF distribution by DPMM cluster\nOverlay: Sublineage/Phasing Relationship')
    plt.legend(title='Annotation', loc='upper right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_vaf_cluster_hist_pro.png", dpi=350)
    plt.close()

def plot_vaf_box_by_cluster(var_df, out_prefix):
    plt.figure(figsize=(10,5))
    sns.violinplot(data=var_df, x='cluster', y='vaf', inner='quartile', palette='Set2', cut=0, linewidth=1.3)
    for ctx, col, marker in zip(['Sublineage', 'PhasedWithSublineage', 'NotPhasedWithSublineage'],
                                ['#D62728', '#1f77b4', '#7F7F7F'],
                                ['o', 's', '^']):
        ctx_vars = var_df[var_df['phasing_context'] == ctx]
        plt.scatter(ctx_vars['cluster'], ctx_vars['vaf'], alpha=0.85, color=col, marker=marker, label=ctx, edgecolor='k', zorder=2, s=60)
    plt.ylabel('VAF')
    plt.title('Cluster-wise VAF distribution\nwith sublineage/phase annotation overlay')
    plt.legend(title='Variant Context', loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cluster_violin_with_context.png", dpi=350)
    plt.close()

def plot_block_summary(cluster_analysis_df, out_prefix):
    plt.figure(figsize=(9,4))
    ind = np.arange(len(cluster_analysis_df))
    bars = plt.bar(ind, cluster_analysis_df['n_blocks'], color="#7AA6C2", alpha=0.75, width=0.6, label="All blocks")
    plt.bar(ind, cluster_analysis_df['num_sublineage_blocks'], color="#D62728", alpha=0.72, width=0.6, label="Sublineage blocks")
    plt.xticks(ind, [str(x) for x in cluster_analysis_df['cluster_label']])
    plt.xlabel('Cluster')
    plt.ylabel('Blocks in Cluster')
    plt.title('Blocks per cluster (red: sublineage-harboring)')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_block_counts_by_cluster.png", dpi=350)
    plt.close()

def plot_vaf_sub_vs_non(cluster_analysis_df, out_prefix):
    records = []
    for idx, row in cluster_analysis_df.iterrows():
        cid = row['cluster_label']
        sub_vafs = safe_split_vaf(row.get('vafs_in_sublineage_blocks', ''))
        nonsub_vafs = safe_split_vaf(row.get('vafs_in_no_sub_blocks', ''))
        for v in sub_vafs:
            records.append({'cluster': cid, 'type': 'sublineage_block', 'vaf': v})
        for v in nonsub_vafs:
            records.append({'cluster': cid, 'type': 'no_sublineage', 'vaf': v})
    df = pd.DataFrame(records)
    if not df.empty:
        plt.figure(figsize=(12,6), dpi=110)
        sns.stripplot(
            data=df, x='cluster', y='vaf', hue='type', 
            palette={'sublineage_block':'#D62728', 'no_sublineage':'#7AA6C2'}, 
            dodge=True, jitter=0.24, alpha=0.7, size=6, linewidth=0.5, edgecolor='k'
        )
        plt.title('VAF distributions: Sublineage vs. Non-Sublineage blocks per cluster')
        plt.ylabel('VAF')
        plt.xlabel('Cluster')
        plt.legend(title="Block Type")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_vaf_sub_vs_non_block.png", dpi=350)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPMM VAF clustering and blocks/sublineage analysis with cluster-consistent splitting')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--blocks', required=True, help='Block assignment file (CSV)')
    parser.add_argument('--sublineage', required=True, help='sublineage varid file (one per line)')
    parser.add_argument('--out_prefix', required=True, help='Output prefix')
    parser.add_argument('--n_init', type=int, default=100)
    args = parser.parse_args()
    variants, varid_vaf = parse_vcf_with_vaf(args.vcf)
    blocks = load_blocks_csv(args.blocks)
    sublineage_variants = load_sublineage_variants(args.sublineage)
    clusters = assign_clusters_dpmms(variants, n_init=args.n_init)
    print("DPMM clustering completed.")
    print("Running block/sublineage analysis and writing cluster segment CSVs...")
    blockwise_analysis(clusters, variants, varid_vaf, blocks, sublineage_variants, args.out_prefix)
    print("Cluster/blockwise CSVs and summary written.")
    var_df = pd.DataFrame([{**v, 'cluster': clusters[i]} for i, v in enumerate(variants)])
    print("Generating journal-quality visualizations...")
    plot_vaf_clusters(var_df, sublineage_variants, blocks, args.out_prefix)
    plot_vaf_box_by_cluster(var_df, args.out_prefix)
    cluster_analysis_df = pd.read_csv(f"{args.out_prefix}_cluster_analysis.csv")
    plot_block_summary(cluster_analysis_df, args.out_prefix)
    plot_vaf_sub_vs_non(cluster_analysis_df, args.out_prefix)
    print("All journal-style plots saved.")
