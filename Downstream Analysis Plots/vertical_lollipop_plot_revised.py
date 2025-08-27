import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
# All annotated variants
all_variants = pd.read_csv('Final_annotated_variants_collated.csv')

# Truly non-sublineage variants (neither sublineage nor phased with sublineage)
truly_nonsub = pd.read_csv('non_sublineage_variants_after_phasing_and_clustering.csv')

# Known sublineage variants
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

# --- Process all variants ---
# Filter for SNVs only
all_variants_snv = all_variants[
    all_variants.apply(lambda row: is_snv(row['Ref'], row['Alt']), axis=1)
].copy()

# Create variant IDs for all variants
all_variants_snv['variant_id'] = all_variants_snv.apply(create_variant_id, axis=1)

# --- Case 1: Remove only known sublineage variants ---
# Keep variants that are NOT in sublineage set
case1_variants = all_variants_snv[
    ~all_variants_snv['variant_id'].isin(sublineage_set)
].copy()

# --- Case 2: Truly non-sublineage variants (already filtered) ---
# Parse variant IDs to get positions
truly_nonsub[['Chrom', 'Pos', 'Ref', 'Alt']] = truly_nonsub['variant_id'].apply(
    lambda v: pd.Series(parse_variant_id(v))
)

# Filter for SNVs only in truly non-sublineage
case2_variants = truly_nonsub[
    truly_nonsub.apply(lambda row: is_snv(row['Ref'], row['Alt']), axis=1)
].copy()
print(len(case2_variants))
case2_variants = case2_variants[~case2_variants['variant_id'].isin(sublineage_set)].copy()
print(len(case2_variants))
# --- Calculate position frequencies ---
def get_position_frequencies(df, sample_col, pos_col='Pos'):
    """Get frequency of samples per genomic position"""
    return df.groupby(pos_col)[sample_col].nunique().to_dict()

# Get frequencies for both cases
case1_freq = get_position_frequencies(case1_variants, 'Sample')
case2_freq = get_position_frequencies(case2_variants, 'sample')

# Get all unique positions and sort them
all_positions = sorted(set(list(case1_freq.keys()) + list(case2_freq.keys())))

# Prepare data for plotting
positions = []
case1_freqs = []
case2_freqs = []

for pos in all_positions:
    positions.append(pos)
    case1_freqs.append(case1_freq.get(pos, 0))
    case2_freqs.append(case2_freq.get(pos, 0))

# --- Calculate cosine similarity ---
if len(case1_freqs) > 0 and any(case1_freqs) and any(case2_freqs):
    # Reshape for sklearn
    case1_vector = np.array(case1_freqs).reshape(1, -1)
    case2_vector = np.array(case2_freqs).reshape(1, -1)
    
    # Calculate cosine similarity
    cos_sim = cosine_similarity(case1_vector, case2_vector)[0][0]
else:
    cos_sim = 0.0  # If one profile is all zeros

# --- Create mirror lollipop plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# Colorblind-friendly colors (from colorbrewer)
color_case1 = '#2166AC'  # Blue
color_case2 = '#762A83'  # Purple

# Case 1: Left side (negative x values but positive display)
ax.barh(positions, [-f for f in case1_freqs], color=color_case1, alpha=0.7, 
        label='Sublineage removed only', height=20)

# Case 2: Right side (positive x values)
ax.barh(positions, case2_freqs, color=color_case2, alpha=0.7, 
        label='Sublineage + phased removed', height=20)

# Add stems for lollipop effect
for pos, freq in zip(positions, case1_freqs):
    if freq > 0:
        ax.plot([-freq, 0], [pos, pos], color=color_case1, linewidth=1.5, alpha=0.8)
        ax.scatter([-freq], [pos], color=color_case1, s=40, zorder=5)

for pos, freq in zip(positions, case2_freqs):
    if freq > 0:
        ax.plot([0, freq], [pos, pos], color=color_case2, linewidth=1.5, alpha=0.8)
        ax.scatter([freq], [pos], color=color_case2, s=40, zorder=5)

# Center line
ax.axvline(0, color='black', linewidth=2, alpha=0.8)

# Styling
ax.set_xlabel('Frequency (Number of Samples)', fontsize=13)
ax.set_ylabel('Genomic Position', fontsize=13)
ax.set_title('SNV Position Frequencies: Sublineage Filtering Comparison', fontsize=15)

# Format x-axis to show positive values on both sides
max_freq = max(max(case1_freqs), max(case2_freqs))
x_ticks = list(range(-max_freq, max_freq + 1, max(1, max_freq // 5)))
x_labels = [str(abs(x)) for x in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

# Y-axis formatting
y_tick_spacing = max(len(positions) // 20, 1)
ax.set_yticks(positions[::y_tick_spacing])

# Legend
ax.legend(loc='lower right', fontsize=12)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.grid(axis='y', alpha=0.2, linestyle=':')

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add cosine similarity annotation
ax.text(0.02, 0.98, f'Cosine Similarity: {cos_sim:.3f}', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
        horizontalalignment='left', verticalalignment='top')

# Add case annotations
ax.text(-max_freq * 0.8, max(positions) * 0.95, 
        f'Case 1: {len([f for f in case1_freqs if f > 0])} positions\n{sum(case1_freqs)} total variants',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

ax.text(max_freq * 0.4, max(positions) * 0.95, 
        f'Case 2: {len([f for f in case2_freqs if f > 0])} positions\n{sum(case2_freqs)} total variants',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='plum', alpha=0.7))

plt.tight_layout()
plt.savefig('mirror_lollipop_snv_frequencies_with_similarity.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print summary statistics ---
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

