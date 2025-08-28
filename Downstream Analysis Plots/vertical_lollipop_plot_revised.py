import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


all_variants = pd.read_csv('Final_annotated_variants_collated.csv')


truly_nonsub = pd.read_csv('non_sublineage_variants_after_phasing_and_clustering.csv')


with open('sublineage_variants.txt') as f:
    sublineage_set = set(line.strip() for line in f if line.strip())

def is_snv(ref, alt):
    """Check if variant is SNV (single nucleotide variant, not indel)"""
    return len(ref) == 1 and len(alt) == 1 and ref.isalpha() and alt.isalpha()

def create_variant_id(row):
    """Create variant ID from chromosome, position, ref, alt"""
    return f"{row['Chrom']}_{row['Pos']}_{row['Ref']}_{row['Alt']}"

def parse_variant_id(vid):
    """Parse variant_id into components"""
    parts = vid.split('_')
    chrom = parts[0]
    pos = int(parts[1])
    ref = parts[2]
    alt = '_'.join(parts[3:]) if len(parts) > 4 else parts[3]
    return chrom, pos, ref, alt


all_variants_snv = all_variants[
    all_variants.apply(lambda row: is_snv(row['Ref'], row['Alt']), axis=1)
].copy()


all_variants_snv['variant_id'] = all_variants_snv.apply(create_variant_id, axis=1)


case1_variants = all_variants_snv[
    ~all_variants_snv['variant_id'].isin(sublineage_set)
].copy()


truly_nonsub[['Chrom', 'Pos', 'Ref', 'Alt']] = truly_nonsub['variant_id'].apply(
    lambda v: pd.Series(parse_variant_id(v))
)


case2_variants = truly_nonsub[
    truly_nonsub.apply(lambda row: is_snv(row['Ref'], row['Alt']), axis=1)
].copy()
print(len(case2_variants))
case2_variants = case2_variants[~case2_variants['variant_id'].isin(sublineage_set)].copy()
print(len(case2_variants))

def get_position_frequencies(df, sample_col, pos_col='Pos'):
    """Get frequency of samples per genomic position"""
    return df.groupby(pos_col)[sample_col].nunique().to_dict()


case1_freq = get_position_frequencies(case1_variants, 'Sample')
case2_freq = get_position_frequencies(case2_variants, 'sample')


all_positions = sorted(set(list(case1_freq.keys()) + list(case2_freq.keys())))


positions = []
case1_freqs = []
case2_freqs = []

for pos in all_positions:
    positions.append(pos)
    case1_freqs.append(case1_freq.get(pos, 0))
    case2_freqs.append(case2_freq.get(pos, 0))


if len(case1_freqs) > 0 and any(case1_freqs) and any(case2_freqs):

    case1_vector = np.array(case1_freqs).reshape(1, -1)
    case2_vector = np.array(case2_freqs).reshape(1, -1)
    
    cos_sim = cosine_similarity(case1_vector, case2_vector)[0][0]
else:
    cos_sim = 0.0  # If one profile is all zeros


fig, ax = plt.subplots(figsize=(12, 8))


color_case1 = '#2166AC'  
color_case2 = '#762A83'  


ax.barh(positions, [-f for f in case1_freqs], color=color_case1, alpha=0.7, 
        label='Sublineage removed only', height=20)


ax.barh(positions, case2_freqs, color=color_case2, alpha=0.7, 
        label='Sublineage + phased removed', height=20)


for pos, freq in zip(positions, case1_freqs):
    if freq > 0:
        ax.plot([-freq, 0], [pos, pos], color=color_case1, linewidth=1.5, alpha=0.8)
        ax.scatter([-freq], [pos], color=color_case1, s=40, zorder=5)

for pos, freq in zip(positions, case2_freqs):
    if freq > 0:
        ax.plot([0, freq], [pos, pos], color=color_case2, linewidth=1.5, alpha=0.8)
        ax.scatter([freq], [pos], color=color_case2, s=40, zorder=5)


ax.axvline(0, color='black', linewidth=2, alpha=0.8)

ax.set_xlabel('Frequency (Number of Samples)', fontsize=13)
ax.set_ylabel('Genomic Position', fontsize=13)
ax.set_title('SNV Position Frequencies: Sublineage Filtering Comparison', fontsize=15)

max_freq = max(max(case1_freqs), max(case2_freqs))
x_ticks = list(range(-max_freq, max_freq + 1, max(1, max_freq // 5)))
x_labels = [str(abs(x)) for x in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

y_tick_spacing = max(len(positions) // 20, 1)
ax.set_yticks(positions[::y_tick_spacing])


ax.legend(loc='lower right', fontsize=12)


ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.grid(axis='y', alpha=0.2, linestyle=':')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0.02, 0.98, f'Cosine Similarity: {cos_sim:.3f}', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
        horizontalalignment='left', verticalalignment='top')

ax.text(-max_freq * 0.8, max(positions) * 0.95, 
        f'Case 1: {len([f for f in case1_freqs if f > 0])} positions\n{sum(case1_freqs)} total variants',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

ax.text(max_freq * 0.4, max(positions) * 0.95, 
        f'Case 2: {len([f for f in case2_freqs if f > 0])} positions\n{sum(case2_freqs)} total variants',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.7))

plt.tight_layout()
plt.savefig('mirror_lollipop_snv_frequencies_with_similarity.png', dpi=300, bbox_inches='tight')
plt.show()

print("=" * 60)
print("MIRROR LOLLIPOP PLOT SUMMARY")
print("=" * 60)
print(f"Total unique positions analyzed: {len(positions)}")
print(f"SNVs only (no indels)")
print()
print("Case 1 - Sublineage removed only:")
print(f"  Positions with variants: {len([f for f in case1_freqs if f > 0])}")
print(f"  Total variant instances: {sum(case1_freqs)}")
print(f"  Max frequency at position: {max(case1_freqs)}")
print()
print("Case 2 - Sublineage + phased removed:")
print(f"  Positions with variants: {len([f for f in case2_freqs if f > 0])}")
print(f"  Total variant instances: {sum(case2_freqs)}")
print(f"  Max frequency at position: {max(case2_freqs)}")
print()
print(f"Cosine similarity between Case 1 and Case 2 profiles: {cos_sim:.3f}")
print("=" * 60)

