import numpy as np
import pysam
from collections import defaultdict
import scipy.stats as stats
from math import log2
from scipy.sparse import dok_matrix, csr_matrix
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import pysam
from collections import defaultdict
from scipy.stats import hypergeom
import numpy as np
import pandas as pd
import pysam
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from collections import defaultdict
from sklearn.preprocessing import normalize
from cdlib import algorithms, evaluation
import argparse


def parse_vcf_with_vaf(vcf_file):
    """Parse VCF and extract variants with VAF"""
    variants = []
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
                variants.append({
                    "varid": vname, "chrom": chrom,
                    "pos": pos, "ref": ref,
                    "alt": alt, "vaf": af
                })
    return variants

def build_sparse_molecule_matrices(bam_file, variants, mapq_min=20, baseq_min=20):
    """Efficiently build sparse molecule (UMI) x variant matrices with quality filtering."""
    pos2vars = defaultdict(list)
    for v in variants:
        key = (v["chrom"], v["pos"])
        pos2vars[key].append(v["varid"])
    V = [v["varid"] for v in variants]
    var_idx = {vid:i for i, vid in enumerate(V)}
    umi_idx = {}
    U = []
    cov_matrix = dok_matrix((0, len(V)), dtype=bool) # will grow
    alt_matrix = dok_matrix((0, len(V)), dtype=bool)
    next_umi = 0

    samfile = pysam.AlignmentFile(bam_file, "rb")

    for read in samfile.fetch():
        if read.is_unmapped or read.mapping_quality < mapq_min:
            continue
        try:
            umi = read.get_tag('RX')
            mi = read.get_tag('MI')
        except KeyError:
            continue

        fam = umi + '_' + str(mi)
        if fam not in umi_idx:
            umi_idx[fam] = next_umi
            U.append(fam)
            next_umi += 1
        fam_idx = umi_idx[fam]
        # Resize sparse matrices dynamically if new UMI
        if fam_idx >= cov_matrix.shape[0]:
            cov_matrix.resize((fam_idx+1, len(V)))
            alt_matrix.resize((fam_idx+1, len(V)))
        rseq = read.query_sequence
        ref_positions = read.get_reference_positions(full_length=True)

        for pos_idx, ref_pos in enumerate(ref_positions):
            if ref_pos is None:
                continue
            key = (read.reference_name, ref_pos + 1)
            if key in pos2vars:
                qual = read.query_qualities
                if pos_idx < len(qual) and qual[pos_idx] < baseq_min:
                    continue
                for vid in pos2vars[key]:
                    v = [v for v in variants if v["varid"] == vid][0]
                    var_i = var_idx[vid]
                    cov_matrix[fam_idx, var_i] = True
                    if pos_idx < len(rseq) and rseq[pos_idx] == v["alt"]:
                        alt_matrix[fam_idx, var_i] = True
    samfile.close()
    return cov_matrix.tocsr(), alt_matrix.tocsr(), U, V

def mutual_information(n00, n01, n10, n11):
    total = n00 + n01 + n10 + n11
    mi = 0
    for x, y, n in [ (0,0,n00), (0,1,n01), (1,0,n10), (1,1,n11) ]:
        if n == 0: continue
        p_xy = n / total
        p_x = (n10 + n11)/total if x==1 else (n00+n01)/total
        p_y = (n01 + n11)/total if y==1 else (n00+n10)/total
        mi += p_xy * log2(p_xy / (p_x * p_y))
    return mi

def normalized_mi(n00, n01, n10, n11):
    total = n00 + n01 + n10 + n11
    p_a1 = (n10 + n11)/total
    p_b1 = (n01 + n11)/total
    h_a = -sum([p*log2(p) for p in [p_a1,1-p_a1] if p > 0])
    h_b = -sum([p*log2(p) for p in [p_b1,1-p_b1] if p > 0])
    mi = mutual_information(n00,n01,n10,n11)
    if min(h_a, h_b) > 0: return mi / min(h_a, h_b)
    return 0

def compute_comprehensive_stats(n00, n01, n10, n11):
    # Build contingency
    contingency = np.array([[n00, n01], [n10, n11]])
    # Conditional probabilities
    p_b_given_a = n11 / (n10+n11) if (n10+n11) > 0 else 0
    p_a_given_b = n11 / (n01+n11) if (n01+n11) > 0 else 0
    # Fisher's test
    _, pval = stats.fisher_exact(contingency)
    n = n00 + n01 + n10 + n11
    mi = mutual_information(n00, n01, n10, n11)
    nmi = normalized_mi(n00, n01, n10, n11)
    pval_hypergeom = hypergeom.sf(n11, n, n10+n11, n01+n11) if n11 > 0 else 1.0
    return dict(p_b_given_a=p_b_given_a, p_a_given_b=p_a_given_b, pval=pval, mi=mi, nmi=nmi, pval_hypergeom=pval_hypergeom)

def meets_phasing_criteria(stats):
    cond_criterion = stats['p_b_given_a'] >= 0.9 and stats['p_a_given_b'] >= 0.9
    sig_criterion = stats['pval'] < 0.001
    hypergeom_criterion = stats['pval_hypergeom'] < 0.001
    mi_criterion = stats['nmi'] >= 0.9
    return sum([cond_criterion, hypergeom_criterion, sig_criterion, mi_criterion]) >= 2

def build_statistical_phasing_graph(cov_matrix, alt_matrix, V, min_molecules=3):
    n_vars = len(V)
    G = nx.Graph()
    for i in range(n_vars):
        G.add_node(V[i])
    edge_stats = {}
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Efficient intersection via CSR rows
            cov_i = cov_matrix[:,i].toarray().flatten()
            cov_j = cov_matrix[:,j].toarray().flatten()
            mask = (cov_i == 1) & (cov_j == 1)
            if np.sum(mask) < min_molecules: continue
            a_states = alt_matrix[mask, i].toarray().flatten()
            b_states = alt_matrix[mask, j].toarray().flatten()
            n00 = np.sum((a_states == 0) & (b_states == 0))
            n01 = np.sum((a_states == 0) & (b_states == 1))
            n10 = np.sum((a_states == 1) & (b_states == 0))
            n11 = np.sum((a_states == 1) & (b_states == 1))
            stats_dict = compute_comprehensive_stats(n00, n01, n10, n11)
            if meets_phasing_criteria(stats_dict):
                G.add_edge(V[i], V[j], **stats_dict)
                edge_stats[(V[i],V[j])] = stats_dict
    return G, edge_stats

def extract_blocks(G):
    return [list(comp) for comp in nx.connected_components(G)]

def plot_edge_stats(edge_stats, out_prefix):
    vals_hyper = [v['pval_hypergeom'] for v in edge_stats.values()]
    vals_nmi = [v['nmi'] for v in edge_stats.values()]
    vals_pval = [-np.log10(max(v['pval'], 1e-20)) for v in edge_stats.values()]
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); sns.histplot(vals_hyper, bins=20); plt.title("Hypergeometric p")
    plt.subplot(1,3,2); sns.histplot(vals_nmi, bins=20); plt.title("Normalized MI")
    plt.subplot(1,3,3); sns.histplot(vals_pval, bins=20); plt.title("-log10 Fisher p")
    plt.tight_layout(); plt.savefig(f"{out_prefix}_phasing_edge_stats.png"); plt.close()

def plot_phase_graph(G, blocks, fname):
    plt.figure(figsize=(14,7))
    pos = nx.spring_layout(G, seed=42)
    cols = sns.color_palette("tab20", n_colors=len(blocks))
    color_dict = {n : c for b,c in zip(blocks,cols) for n in b}
    node_colors = [color_dict.get(n, (0.5,0.5,0.5)) for n in G.nodes()]
    nx.draw_networkx(G, pos, node_color=node_colors, with_labels=True, node_size=360)
    plt.title("Statistical Phasing Network")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_block_sizes(blocks, out_prefix):
    plt.figure(figsize=(7,4))
    sns.countplot([len(b) for b in blocks], color="slateblue")
    plt.title("Phase Block Size Distribution")
    plt.xlabel("Block Size (#Variants)")
    plt.ylabel("Block Count")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_block_size_histogram.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse matrix statistical variant phasing for viral/bulk sequencing")
    parser.add_argument('--input_vcf', required=True, help="Input VCF filename")
    parser.add_argument('--input_bam', required=True, help="Input BAM filename (UMI consensus)")
    parser.add_argument('--min_molecules', type=int, default=3, help='Minimum molecules required for correlation analysis')
    parser.add_argument('--mapq_min', type=int, default=20, help='Minimum mapping quality (default: 20)')
    parser.add_argument('--baseq_min', type=int, default=20, help='Minimum base quality (default: 20)')
    parser.add_argument('--output_prefix', type=str, default='phasing_analysis', help='Prefix for all output')
    parser.add_argument('--output', required=True, help='Output CSV path for blocks table')
    args = parser.parse_args()

    variants = parse_vcf_with_vaf(args.input_vcf)
    cov_mtx, alt_mtx, U, V = build_sparse_molecule_matrices(
        args.input_bam, variants, 
        args.mapq_min, 
        args.baseq_min
    )
    G, edge_stats = build_statistical_phasing_graph(
        cov_mtx, alt_mtx, V, args.min_molecules
    )
    blocks = extract_blocks(G)
    plot_edge_stats(edge_stats, args.output_prefix)
    plot_phase_graph(G, blocks, fname=f"{args.output_prefix}_phasing_graph.png")
    plot_block_sizes(blocks, args.output_prefix)
    # Pad and write block table
    max_length = max(len(block) for block in blocks)
    block_df = pd.DataFrame(
        [b + ['']*(max_length-len(b)) for b in blocks],
        columns=[f"var_{i}" for i in range(max_length)]
    ).T
    block_df.columns = [f"block_{i}" for i in range(len(blocks))]
    block_df.to_csv(args.output, index=False)
    print(f"Phase blocks saved to {args.output}. See graphs starting with {args.output_prefix}_*.png")

