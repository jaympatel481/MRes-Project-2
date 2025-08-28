import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Phylo import draw
import warnings
warnings.filterwarnings('ignore')

def parse_vcf_variants(vcf_file, sample_name):
    
    variants = {}
    
    with open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt_alleles = fields[4].split(',')
            
            # Skip indels - only keep SNVs
            if len(ref) != 1:
                continue
                
            info = fields[8].split(':')
            vals = fields[9].split(':')
            
            try:
                af = float(vals[info.index('AF')].split(',')[0])
            except Exception:
                af = 0.0
                
            for alt in alt_alleles:
                # Skip indels
                if len(alt) != 1:
                    continue
                    
                vname = f"{chrom}_{pos}_{ref}_{alt}"
                variants[vname] = af
                
    return variants

def count_sublineage_variants_per_sample(samples_file, sublineage_file):
    """Count the number of known sublineage variants present in each sample"""
    
    
    with open(samples_file) as f:
        sample_names = [line.strip() for line in f if line.strip()]
    
    # Load sublineage variants
    with open(sublineage_file) as f:
        sublineage_vars = set(line.strip() for line in f if line.strip())
    
    sublineage_counts = {}
    
    for sample in sample_names:
        vcf_path = os.path.join(sample, "hpv_somatic_snvs.vcf")
        if os.path.exists(vcf_path):
            variants = parse_vcf_variants(vcf_path, sample)
            # Count how many sublineage variants are present in this sample
            sample_sublineage_count = len([v for v in variants.keys() if v in sublineage_vars])
            sublineage_counts[sample] = sample_sublineage_count
        else:
            sublineage_counts[sample] = 0
    
    return sublineage_counts

def load_truly_nonsublineage_variants(sample, sublineage_variants):
    
    
    
    vcf_path = os.path.join(sample, "hpv_somatic_snvs.vcf")
    block_csv = os.path.join(sample, "UMI_Phased_Blocks.csv")
    
    if not (os.path.exists(vcf_path) and os.path.exists(block_csv)):
        return set()
    
    
    vaf_dict = {}
    with open(vcf_path) as vf:
        for line in vf:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            ref = fields[3]
            alt_alleles = fields[4].split(",")
            
            # Skip indels
            if len(ref) != 1:
                continue
                
            info = fields[8].split(":")
            vals = fields[9].split(":")
            try:
                af = float(vals[info.index("AF")].split(",")[0])
            except Exception:
                af = 0.0
            for alt in alt_alleles:
                # Skip indels
                if len(alt) != 1:
                    continue
                varid = f"{chrom}_{pos}_{ref}_{alt}"
                vaf_dict[varid] = af

    
    blocks = []
    try:
        block_df = pd.read_csv(block_csv)
        for col in block_df.columns:
            col_variants = [v for v in block_df[col].dropna().astype(str) if v.strip()]
            blocks.append(set(col_variants))
    except:
        blocks = []

    # Find sublineage variants in this sample
    sample_sublineage_vars = set([v for v in vaf_dict if v in sublineage_variants])
    
    # Find blocks that contain sublineage variants
    blocks_with_sub = set()
    for bi, blk in enumerate(blocks):
        if blk & sample_sublineage_vars:
            blocks_with_sub.add(bi)
    
    # Determine truly non-sublineage variants
    truly_non_sub_variants = set()
    for varid in vaf_dict:
        if varid in sample_sublineage_vars:
            continue  # Skip sublineage variants
        
        # Check if this variant is in any block that contains sublineage variants
        is_phased_with_sublineage = False
        for bi, blk in enumerate(blocks):
            if varid in blk and bi in blocks_with_sub:
                is_phased_with_sublineage = True
                break
        
        if not is_phased_with_sublineage:
            truly_non_sub_variants.add(varid)
    
    return truly_non_sub_variants

def create_vaf_matrix(samples_file, sublineage_file=None):
    
    
    
    with open(samples_file) as f:
        sample_names = [line.strip() for line in f if line.strip()]
    
    
    sublineage_vars = set()
    if sublineage_file:
        with open(sublineage_file) as f:
            sublineage_vars = set(line.strip() for line in f if line.strip())
    
    
    all_variants = set()
    sample_data = {}
    truly_non_sub_variants_per_sample = {}
    all_truly_non_sub_variants = set()
    
    print("Parsing VCF files and identifying variant categories...")
    for sample in sample_names:
        vcf_path = os.path.join(sample, "hpv_somatic_snvs.vcf")
        if os.path.exists(vcf_path):
            variants = parse_vcf_variants(vcf_path, sample)
            sample_data[sample] = variants
            all_variants.update(variants.keys())
            
            
            truly_non_sub = load_truly_nonsublineage_variants(sample, sublineage_vars)
            truly_non_sub_variants_per_sample[sample] = truly_non_sub
            all_truly_non_sub_variants.update(truly_non_sub)
            
            print(f"  {sample}: {len(variants)} total SNVs, {len(truly_non_sub)} truly non-sublineage")
        else:
            print(f"  Warning: VCF not found for {sample}")
            truly_non_sub_variants_per_sample[sample] = set()
    
    all_variants = sorted(list(all_variants))
    valid_samples = [s for s in sample_names if s in sample_data]
    
    print(f"\nTotal unique SNVs across all samples: {len(all_variants)}")
    print(f"Valid samples: {len(valid_samples)}")
    
    # VAF matrix - samples as rows, variants as columns
    vaf_matrix = pd.DataFrame(0.0, index=valid_samples, columns=all_variants)
    
    for sample in valid_samples:
        for variant in all_variants:
            vaf_matrix.loc[sample, variant] = sample_data[sample].get(variant, 0.0)
    
    
    non_sub_variants = [v for v in all_variants if v not in sublineage_vars]
    non_sub_matrix = vaf_matrix[non_sub_variants].copy()
    
    
    all_truly_non_sub_variants = sorted(list(all_truly_non_sub_variants))
    truly_non_sub_matrix = pd.DataFrame(0.0, index=valid_samples, columns=all_truly_non_sub_variants)
    
    for sample in valid_samples:
        sample_truly_non_sub = truly_non_sub_variants_per_sample[sample]
        for variant in all_truly_non_sub_variants:
            if variant in sample_truly_non_sub and variant in sample_data[sample]:
                truly_non_sub_matrix.loc[sample, variant] = sample_data[sample][variant]
    
    print(f"Non-sublineage SNVs (not in sublineage list): {len(non_sub_variants)}")
    print(f"Truly non-sublineage SNVs (not sublineage AND not phased): {len(all_truly_non_sub_variants)}")
    
    return vaf_matrix, non_sub_matrix, truly_non_sub_matrix, valid_samples

def calculate_vaf_distance(matrix, method='euclidean'):
    
    if method == 'euclidean':
        distances = pairwise_distances(matrix.values, metric='euclidean')
    else:
        raise ValueError(f"Only euclidean distance supported in this version")
    
    return distances

def build_phylogenetic_tree(distance_matrix, sample_names, method='nj'):
    
    
    
    n_samples = len(sample_names)
    bio_matrix = []
    
    for i in range(n_samples):
        row = []
        for j in range(i + 1):
            row.append(distance_matrix[i][j])
        bio_matrix.append(row)
    
    dm = DistanceMatrix(names=sample_names, matrix=bio_matrix)
    
    
    constructor = DistanceTreeConstructor()
    
    if method == 'nj':
        tree = constructor.nj(dm)
    else:
        raise ValueError(f"Only neighbor joining supported in this version")
    
    return tree

def plot_colored_tree(tree, title, filename, sublineage_counts, figsize=(12, 8)):
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    norm = Normalize(vmin=0, vmax=40)
    cmap = cm.get_cmap('plasma')
    
    
    def color_terminals(tree):
        
        terminals = tree.get_terminals()
        for terminal in terminals:
            if terminal.name:
                count = sublineage_counts.get(terminal.name, 0)
                count_capped = min(count, 40)
                color = cmap(norm(count_capped))
                # Convert to hex for BioPython
                terminal.color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0]*255), int(color[1]*255), int(color[2]*255)
                )
    
    
    color_terminals(tree)
    
    
    Phylo.draw(tree, axes=ax, do_show=False, 
               branch_labels=None, 
               label_func=lambda x: "",  # No text labels
               show_confidence=False)
    
    
    lines = ax.get_lines()
    terminals = tree.get_terminals()
    
    
    ax.clear()
    
    
    import matplotlib.patches as patches
    from matplotlib.collections import LineCollection
    
    
    Phylo.draw(tree, axes=ax, do_show=False, 
               branch_labels=None, 
               label_func=lambda x: "",
               show_confidence=False)
    
    
    
    all_lines = ax.get_lines()
    
    
    
    
    
    terminals_list = tree.get_terminals()
    n_terminals = len(terminals_list)
    
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    
    
    y_positions = np.linspace(ylim[0], ylim[1], n_terminals)
    x_position = xlim[1] * 0.99  # Near the right edge
    
    
    for i, terminal in enumerate(terminals_list):
        if terminal.name:
            count = sublineage_counts.get(terminal.name, 0)
            count_capped = min(count, 40)
            color = cmap(norm(count_capped))
            
            
            circle = patches.Circle((x_position, y_positions[i]), 
                                  radius=(xlim[1] - xlim[0]) * 0.015,  
                                  facecolor=color, 
                                  edgecolor='black',
                                  linewidth=0.5,
                                  zorder=10)
            ax.add_patch(circle)
    
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Number of Known Sublineage Variants', fontsize=12)
    cbar.set_ticks([0, 10, 20, 30, 40])
    cbar.set_ticklabels(['0', '10', '20', '30', '40+'])
    
    
    ax.text(0.02, 0.98, 'Branch lengths represent\nEuclidean VAF distances\n\nLeaf colors: # sublineage variants', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    
    
    
    samples_file = "samples_phasing_clustering_success.txt"
    sublineage_file = "sublineage_variants.txt"
    output_prefix = "phylogenetic_analysis"
    
    print("=== HPV Phylogenetic Analysis (Colored Leaf Tips Version) ===\n")
    
    
    print("0. Counting sublineage variants per sample...")
    sublineage_counts = count_sublineage_variants_per_sample(samples_file, sublineage_file)
    print("   Sublineage variant counts calculated.")
    
    
    print("1. Creating VAF matrices...")
    all_vaf_matrix, non_sub_matrix, truly_non_sub_matrix, sample_names = create_vaf_matrix(
        samples_file, sublineage_file)
    
    
    all_vaf_matrix.to_csv(f"{output_prefix}_all_variants_vaf_matrix.csv")
    non_sub_matrix.to_csv(f"{output_prefix}_non_sublineage_vaf_matrix.csv")
    truly_non_sub_matrix.to_csv(f"{output_prefix}_truly_non_sublineage_vaf_matrix.csv")
    print("   VAF matrices saved.")
    
    
    matrix_configs = [
        ('all_variants', all_vaf_matrix, 'All Variants'),
        ('non_sublineage', non_sub_matrix, 'Sublineage Variants Removed'),
        ('truly_non_sublineage', truly_non_sub_matrix, 'Truly Non-sublineage Variants Only')
    ]
    
    
    dist_method = 'euclidean'
    tree_method = 'nj'
    
    for matrix_name, matrix, matrix_desc in matrix_configs:
        print(f"\n--- Processing {matrix_desc} ---")
        
        
        if matrix.shape[1] == 0:
            print(f"   Skipping {matrix_desc}: No variants found")
            continue
        
        print(f"2. Building tree using {dist_method} distance and {tree_method.upper()}...")
        
        
        distances = calculate_vaf_distance(matrix, method=dist_method)
        
        
        tree = build_phylogenetic_tree(distances, sample_names, method=tree_method)
        
        
        plot_colored_tree(
            tree,
            f"Phylogenetic Tree - {matrix_desc}\n(Neighbor Joining, Euclidean distance)",
            f"{output_prefix}_{matrix_name}_{dist_method}_{tree_method}_colored_tips.png",
            sublineage_counts
        )
        
        
        with open(f"{output_prefix}_{matrix_name}_{dist_method}_{tree_method}.nwk", 'w') as f:
            Phylo.write(tree, f, 'newick')
        
        print(f"   Tree saved: {output_prefix}_{matrix_name}_{dist_method}_{tree_method}_colored_tips.png")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total samples analyzed: {len(sample_names)}")
    print(f"Total SNVs (all): {all_vaf_matrix.shape[1]}")
    print(f"Total SNVs (non-sublineage): {non_sub_matrix.shape[1]}")
    print(f"Total SNVs (truly non-sublineage): {truly_non_sub_matrix.shape[1]}")
    print(f"\nGenerated 3 phylogenetic trees with colored leaf tips:")
    print("1. All variants - shows overall sample relationships")
    print("2. Sublineage variants removed - excludes known sublineage-defining variants")  
    print("3. Truly non-sublineage only - excludes both sublineage and phased variants")
    print(f"\nLeaf tip colors represent number of known sublineage variants per sample (YlGn colormap, 0-40+ range)")

if __name__ == "__main__":
    main()

