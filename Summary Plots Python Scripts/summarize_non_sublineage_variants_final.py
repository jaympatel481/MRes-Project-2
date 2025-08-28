import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

samples_txt = "samples_phasing_clustering_success.txt"
sublineage_file = "sublineage_variants.txt"

variant_records = []

with open(sublineage_file) as f:
    sublineage_set = set(line.strip() for line in f if line.strip())

with open(samples_txt) as sf:
    sample_dirs = [line.strip() for line in sf if line.strip()]

for sample in tqdm(sample_dirs):
    
    vcf_path = os.path.join(sample, "hpv_somatic_snvs.vcf")
    dpmm_analysis_path = os.path.join(sample, "dpmm_res_cluster_analysis.csv")

    if not (os.path.exists(vcf_path) and os.path.exists(dpmm_analysis_path)):
        continue

    
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
            info = fields[8].split(":")
            vals = fields[9].split(":")
            try:
                af = float(vals[info.index("AF")].split(",")[0])
            except Exception:
                af = 0.0
            for alt in alt_alleles:
                varid = f"{chrom}_{pos}_{ref}_{alt}"
                vaf_dict[varid] = af

    
    dpmm_df = pd.read_csv(dpmm_analysis_path)
    for i, row in dpmm_df.iterrows():
        cluster_label = row['cluster_label']
        if pd.isna(row.get('variants_not_in_sublineage_blocks', '')):
            continue
        # Get the list of variants
        ns_list = [v for v in str(row['variants_not_in_sublineage_blocks']).split(';') if v.strip()]
        for varid in ns_list:
            if varid in vaf_dict:
                variant_records.append({
                    "variant_id": varid,
                    "sample": sample,
                    "VAF": vaf_dict[varid]
                })


df = pd.DataFrame(variant_records)


sample_counts = df.groupby('variant_id')['sample'].nunique().reset_index()
sample_counts = sample_counts.rename(columns={"sample": "Sample_Count"})

valid_vars = set(sample_counts['variant_id'])
df = df[df['variant_id'].isin(valid_vars)]


variant_order = df.groupby('variant_id')['VAF'].median().sort_values().index.tolist()
df['variant_id'] = pd.Categorical(df['variant_id'], categories=variant_order, ordered=True)
sample_counts['variant_id'] = pd.Categorical(sample_counts['variant_id'], categories=variant_order, ordered=True)


fig, axes = plt.subplots(
    1, 2, figsize=(6, len(variant_order) * 0.11),
    sharey=True, gridspec_kw={'width_ratios': [3, 1]}
)


sns.stripplot(
    data=df,
    x='VAF',
    y='variant_id',
    ax=axes[0],
    jitter=0.2,
    alpha=0.7,
    marker='o',
    s=5,
    color='steelblue',
    order=variant_order
)
axes[0].set_xlabel('Variant Allele Frequency (VAF)', fontsize=12)
axes[0].set_ylabel('')
axes[0].tick_params(axis='y', which='major', labelsize=0)
axes[0].tick_params(axis='x', which='major', labelsize=10)
axes[0].grid(axis='x', linestyle=':', alpha=0.7)
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].spines['left'].set_linewidth(2)
axes[0].spines['bottom'].set_linewidth(2)
axes[0].spines['left'].set_color('black')
axes[0].spines['bottom'].set_color('black')


sns.barplot(
    data=sample_counts,
    x='Sample_Count',
    y='variant_id',
    ax=axes[1],
    color='darkorange',
    edgecolor='black',
    order=variant_order
)
axes[1].set_xlabel('Number of Samples', fontsize=12)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', which='major', labelsize=0)
axes[1].tick_params(axis='x', which='major', labelsize=10)
axes[1].grid(axis='x', linestyle=':', alpha=0.7)
axes[1].spines[['top', 'right']].set_visible(False)
axes[1].spines['left'].set_linewidth(2)
axes[1].spines['bottom'].set_linewidth(2)
axes[1].spines['left'].set_color('black')
axes[1].spines['bottom'].set_color('black')


num_variants = len(variant_order)
print(num_variants)
for i in range(num_variants):
    axes[0].axhline(y=i, color='black', linestyle=':', linewidth=0.7, alpha=0.7)
    axes[1].axhline(y=i, color='black', linestyle=':', linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.savefig("summary_non_sublineage_variants_after_phase_cluster.png",dpi=300)
plt.show()

