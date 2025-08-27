import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- Load cohort/sample mapping ---
cohort_map = pd.read_csv('cohort_id_samples.csv')
cohort_dict = dict(zip(cohort_map['Sample'], cohort_map['Cohort']))

def get_group(sample):
    cohort = cohort_dict.get(sample, '')
    if cohort.startswith('TIN'):
        return 'TIN'
    elif cohort.startswith('SR'):
        return 'SR'
    else:
        return 'Other'

# --- Function to process variant position for lollipop plot ---
def get_variant_pos(row, chrom_col='Chrom', pos_col='Pos'):
    # Ensure int for sorting, or use string if need be
    chrom = row[chrom_col]
    pos = int(row[pos_col])
    return f"{chrom}_{pos}"

# --- Enhanced Lollipop Plot function with cosine similarity ---
def lollipop_plot_with_similarity(var_df, sample_col='Sample', chrom_col='Chrom', pos_col='Pos',
                                plot_title='All variants', filename='lollipop_all.png'):
    # Get sample group
    var_df['Group'] = var_df[sample_col].apply(get_group)
    var_df['pos_key'] = var_df.apply(get_variant_pos, axis=1, chrom_col=chrom_col, pos_col=pos_col)
    
    # Count samples per position/group
    pos_TIN = var_df[var_df['Group']=='TIN'].groupby('pos_key')[sample_col].nunique()
    pos_SR  = var_df[var_df['Group']=='SR'].groupby('pos_key')[sample_col].nunique()
    
    # Get all unique positions and sort them
    unique_positions = sorted(set(list(pos_TIN.index) + list(pos_SR.index)), 
                             key=lambda x: int(x.split('_')[1]))
    
    x = []
    y_TIN = []
    y_SR = []
    tin_counts = []  # For cosine similarity calculation
    sr_counts = []   # For cosine similarity calculation
    
    for pk in unique_positions:
        x.append(int(pk.split('_')[1]))
        tin_count = pos_TIN.get(pk, 0)
        sr_count = pos_SR.get(pk, 0)
        
        y_TIN.append(tin_count)
        y_SR.append(-sr_count)  # negative for below axis
        
        # Store actual counts (not negative) for similarity calculation
        tin_counts.append(tin_count)
        sr_counts.append(sr_count)
    
    # Calculate cosine similarity
    if len(tin_counts) > 0 and any(tin_counts) and any(sr_counts):
        # Reshape for sklearn
        tin_vector = np.array(tin_counts).reshape(1, -1)
        sr_vector = np.array(sr_counts).reshape(1, -1)
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity(tin_vector, sr_vector)[0][0]
    else:
        cos_sim = 0.0  # If one profile is all zeros
    
    # Plot
    fig, ax = plt.subplots(figsize=(13, 4))
    
    # Lollipops above (TIN)
    ax.stem(x, y_TIN, linefmt='C0-', markerfmt='C0o', basefmt=" ", 
            label='TIN')
    
    # Lollipops below (SR)
    ax.stem(x, y_SR, linefmt='C1-', markerfmt='C1o', basefmt=" ", 
            label='SR')
    
    # Horizontal line at y=0
    ax.axhline(0, color='black', linewidth=2)
    
    # Axis labels and title
    ax.set_xlabel('Genome Position', fontsize=13)
    ax.set_ylabel('Number of Samples', fontsize=13)
    ax.set_title(plot_title, fontsize=15)
    
    # X-axis ticks
    xtick_spacing = max(len(x)//15, 1)
    ax.set_xticks(x[::xtick_spacing])
    ax.tick_params(axis='x', labelrotation=90)
    
    # Legend
    ax.legend(["TIN (above)", "SR (below)"], loc='upper left', fontsize=12)
    
    # Add cosine similarity annotation
    # Position the text in the upper right area
    ax.text(0.98, 0.95, f'Cosine Similarity: {cos_sim:.3f}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
            horizontalalignment='right', verticalalignment='top')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {filename}")
    print(f"Cosine similarity between TIN and SR profiles: {cos_sim:.3f}")
    print(f"Number of mutation positions analyzed: {len(unique_positions)}")
    print(f"TIN mutations at {sum(1 for c in tin_counts if c > 0)} positions")
    print(f"SR mutations at {sum(1 for c in sr_counts if c > 0)} positions")
    print("-" * 50)

# --- 1. All variants plot ---
print("Processing all variants...")
allvars = pd.read_csv('Final_annotated_variants_collated.csv')
lollipop_plot_with_similarity(allvars, sample_col='Sample', chrom_col='Chrom', pos_col='Pos',
                             plot_title='All variants: Genome position vs #samples (TIN above, SR below)',
                             filename='lollipop_all_with_similarity.png')

# --- 2. Truly non-sublineage variants only ---
print("Processing truly non-sublineage variants...")
nonsub = pd.read_csv('non_sublineage_variants_after_phasing_and_clustering.csv')

# Parse variant_id into Chrom, Pos
def parse_variant_id(vid):
    # HPV16_5535_C_A or similar
    parts = vid.split('_')
    chrom = parts[0]
    pos = parts[1]
    return chrom, pos

nonsub[['Chrom','Pos']] = nonsub['variant_id'].apply(lambda v: pd.Series(parse_variant_id(v)))
lollipop_plot_with_similarity(nonsub, sample_col='sample', chrom_col='Chrom', pos_col='Pos',
                             plot_title='Truly non-sublineage SNVs: Genome position vs #samples (TIN above, SR below)',
                             filename='lollipop_non_sublineage_with_similarity.png')

