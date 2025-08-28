import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_vaf = pd.read_csv("non_sublineage_variants_after_phasing_and_clustering.csv") # Must have 'variant_id', 'sample', 'VAF'
df_annot = pd.read_csv("Final_annotated_variants_collated.csv")  # Must have 'Sample', 'Chrom', 'Pos', 'Ref', 'Alt' and effect columns


def parse_variant_id(vid):
    tokens = vid.split("_")
    chrom = tokens[0]
    pos = tokens[1]
    ref = tokens[2]
    alt = "_".join(tokens[3:]) if len(tokens) > 4 else tokens[3]
    return chrom, pos, ref, alt

var_info = df_vaf['variant_id'].apply(parse_variant_id)
df_vaf[['Chrom', 'Pos', 'Ref', 'Alt']] = pd.DataFrame(var_info.tolist(), index=df_vaf.index)
df_vaf['Pos'] = df_vaf['Pos'].astype(str)
df_annot['Pos'] = df_annot['Pos'].astype(str)


merged = pd.merge(
    df_vaf,
    df_annot,
    left_on=['sample', 'Chrom', 'Pos', 'Ref', 'Alt'],
    right_on=['Sample', 'Chrom', 'Pos', 'Ref', 'Alt'],
    how='left'
)


snv_mask = (merged['Ref'].str.len() == 1) & (merged['Alt'].str.len() == 1)
merged = merged[snv_mask].copy()


effect_cols = ['E1', 'E2', 'E4', 'E5', 'E6', 'E7', 'L1', 'L2', 'E1^E4']

def classify_variant(df, effect_cols):
    effect_map = {}
    group = df.groupby('variant_id')
    for varid, subdf in group:
        found_non_syn = False
        found_syn = False
        
        for _, row in subdf.iterrows():
            effects = [str(row.get(gene,'')) for gene in effect_cols]
            if any('non_synonymous' in eff or 'missense_variant' in eff or 'stop_lost' in eff or 'stop_gained' in eff or 'start_lost' in eff or 'start_gained' in eff for eff in effects):
                found_non_syn = True
                break  # Non-synonymous checked first as priority
            elif any('synonymous' in eff for eff in effects):
                found_syn = True
        if found_non_syn:
            effect_map[varid] = 'nonsynonymous'
        elif found_syn:
            effect_map[varid] = 'synonymous'
        else:
            effect_map[varid] = 'other'
    return effect_map

effect_map = classify_variant(merged, effect_cols)
merged['effect_class'] = merged['variant_id'].map(effect_map)


palette = {
    'nonsynonymous': '#D73027',     
    'synonymous':    '#1A9850',     
    'other':         '#636363'      
}
label_dict = {
    'nonsynonymous': 'Non-synonymous',
    'synonymous': 'Synonymous',
    'other': 'Other'
}


variant_sample_counts = merged.groupby('variant_id')['sample'].nunique().reset_index()
variant_sample_counts = variant_sample_counts.rename(columns={'sample': 'Sample_Count'})
variant_sample_counts = variant_sample_counts.sort_values('Sample_Count', ascending=False)


variant_top = variant_sample_counts.head(50)
variant_order = variant_top['variant_id'].tolist()


merged_top = merged[merged['variant_id'].isin(variant_order)].copy()
merged_top['variant_id'] = pd.Categorical(merged_top['variant_id'], categories=variant_order, ordered=True)
variant_top['variant_id'] = pd.Categorical(variant_top['variant_id'], categories=variant_order, ordered=True)
merged_top['VAF'] = merged_top['VAF_x'] if 'VAF_x' in merged_top.columns else merged_top['VAF']


fig, axes = plt.subplots(
    1, 2, figsize=(6, len(variant_order) * 0.13),
    sharey=True, gridspec_kw={'width_ratios': [3, 1]}
)

sns.stripplot(
    data=merged_top,
    x='VAF',
    y='variant_id',
    ax=axes[0],
    jitter=0.2,
    alpha=0.7,
    marker='o',
    s=5,
    hue='effect_class',
    palette=palette,
    order=variant_order,
    legend=False
)
axes[0].set_xlabel('Variant Allele Frequency (VAF)', fontsize=12)
axes[0].set_ylabel('')
axes[0].tick_params(axis='y', which='major', labelsize=6)
axes[0].tick_params(axis='x', which='major', labelsize=10)
axes[0].grid(axis='x', linestyle=':', alpha=0.7)
axes[0].spines[['top','right']].set_visible(False)
axes[0].spines['left'].set_linewidth(2)
axes[0].spines['bottom'].set_linewidth(2)
axes[0].spines['left'].set_color('black')
axes[0].spines['bottom'].set_color('black')

sns.barplot(
    data=variant_top,
    x='Sample_Count',
    y='variant_id',
    ax=axes[1],
    color='#FDAE61',
    edgecolor='black',
    order=variant_order
)
axes[1].set_xlabel('Number of Samples', fontsize=12)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', which='major', labelsize=0)
axes[1].tick_params(axis='x', which='major', labelsize=10)
axes[1].grid(axis='x', linestyle=':', alpha=0.7)
axes[1].spines[['top','right']].set_visible(False)
axes[1].spines['left'].set_linewidth(2)
axes[1].spines['bottom'].set_linewidth(2)
axes[1].spines['left'].set_color('black')
axes[1].spines['bottom'].set_color('black')

for i in range(len(variant_order)):
    axes[0].axhline(y=i, color='black', linestyle=':', linewidth=0.7, alpha=0.7)
    axes[1].axhline(y=i, color='black', linestyle=':', linewidth=0.7, alpha=0.7)


from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='none', label=label_dict[k], markerfacecolor=palette[k], markeredgecolor='black', markersize=7, alpha=0.95)
    for k in ['nonsynonymous', 'synonymous', 'other']
]
axes[0].legend(handles=handles, title="Functional Effect", loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig("summary_non_sublineage_snvs_with_annotation.png", dpi=300)
plt.show()

